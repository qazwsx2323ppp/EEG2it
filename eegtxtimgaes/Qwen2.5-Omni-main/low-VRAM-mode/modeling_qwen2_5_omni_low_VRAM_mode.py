import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
from torch import nn 
# EEGEncoder not used here; external EEG embeddings are expected
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import  is_flash_attn_available
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_torch_flex_attn_available,
    logging,
    replace_return_docstrings,
)
from transformers.utils.hub import cached_file
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniConfig,
    Qwen2_5OmniTextConfig,
    Qwen2_5OmniThinkerConfig,

)


if is_flash_attn_2_available():
    from flash_attn.flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func
    from flash_attn.layers.rotary import apply_rotary_emb
else:
    flash_attn_varlen_func = None
    apply_rotary_emb = None


if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask

    from transformers.integrations.flex_attention import make_flex_block_causal_mask



logger = logging.get_logger(__name__)


class Qwen2RMSNorm(nn.Module):
    ## Root Mean Square Normalization (均方根归一化) 层。
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm #均方根归一化
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


Qwen2_5Omni_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Qwen2_5OmniConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Qwen2_5Omni Model outputting raw hidden-states without any specific head on top.",
    Qwen2_5Omni_START_DOCSTRING,
)
class Qwen2_5OmniPreTrainedModel(PreTrainedModel):
    #Qwen2.5-Omni 模型的 预训练模型基类 (Base Class) 
    #权重管理 ：负责模型的权重加载 ( from_pretrained )、保存 ( save_pretrained ) 和下载。
    #配置关联 ：通过 config_class = Qwen2_5OmniConfig 将模型与配置文件关联起来。

    config_class = Qwen2_5OmniConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2_5OmniDecoderLayer", "Qwen2_5OmniVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True

    def _init_weights(self, module):
        # important: this ported version of Qwen2.5OmniThinker isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed
        std = self.config.initializer_range if hasattr(self.config, "initializer_range") else 0.02

        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv3d, nn.ConvTranspose1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        elif isinstance(module, Qwen2RMSNorm):
            module.weight.data.fill_(1.0)

'''
预训练与通用基类：Qwen2_5OmniPreTrainedModel、
Qwen2_5OmniPreTrainedModelForConditionalGeneration（负责权重与生成流程的通用逻辑）
'''
class Qwen2_5OmniPreTrainedModelForConditionalGeneration(Qwen2_5OmniPreTrainedModel):
    #Qwen2.5-Omni 模型的 预训练模型基类 (Base Class) ，用于 条件生成任务（Conditional Generation） 。
    #主要是为了服务于 条件生成任务（Conditional Generation） ，特别是处理注意力掩码（Attention Mask）的逻辑。
    def _prepare_4d_causal_attention_mask_with_cache_position(
        self,
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        min_dtype: float,
        cache_position: torch.Tensor,
        batch_size: int,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            min_dtype (`float`):
                The minimum value representable with the dtype `dtype`.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask

    def get_llm_pos_ids_for_vision(
        self,
        start_idx: int,
        vision_idx: int,
        spatial_merge_size: int,
        t_index: List[int],
        grid_hs: List[int],
        grid_ws: List[int],
    ):
        llm_pos_ids_list = []
        llm_grid_h = grid_hs[vision_idx] // spatial_merge_size
        llm_grid_w = grid_ws[vision_idx] // spatial_merge_size
        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(len(t_index), -1, llm_grid_w).flatten()
        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(len(t_index), llm_grid_h, -1).flatten()
        t_index = torch.Tensor(t_index).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten().long()
        _llm_pos_ids = torch.stack([t_index, h_index, w_index])
        llm_pos_ids_list.append(_llm_pos_ids + start_idx)  # + 1 ) # 12.09 by malinhan
        llm_pos_ids = torch.cat(llm_pos_ids_list, dim=1)
        return llm_pos_ids

    def get_chunked_index(
        self, token_indices: torch.Tensor, tokens_per_chunk: int, remove_index: int
    ) -> list[tuple[int, int]]:
        """
        Splits token index list into chunks based on token value ranges.

        Given a list of token indices, returns a list of (start, end) index tuples representing
        slices of the list where the token values fall within successive ranges of `t_ntoken_per_chunk`.

        For example, if `t_ntoken_per_chunk` is 1000, the function will create chunks such that:
        - the first chunk contains token values < 1000,
        - the second chunk contains values >= 1000 and < 2000, and so on.

        Parameters:
            token_indices (`List[int]`): A monotonically increasing list of token index values.
            t_ntoken_per_chunk (`int`): Number of tokens per chunk (used as the chunk size threshold).
            remove_index (`int`) An index id to subtract from `token_indices` before chunking

        Returns:
            `List[Tuple[int, int]]`: A list of tuples, each representing the start (inclusive)
                                and end (exclusive) indices of a chunk in `token_indices`.
        """

        def _iter():
            i, start_idx = 0, 0  # skip bos token
            current_chunk = 1
            while i < len(token_indices):  # skip eos token
                if token_indices[0][i] - remove_index >= current_chunk * tokens_per_chunk:
                    yield (start_idx, i)
                    start_idx = i
                    current_chunk += 1
                i += 1
            yield (start_idx, token_indices.shape[1])

        return list(_iter())

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_audio_in_video: bool = False,
        audio_seqlens: Optional[torch.LongTensor] = None,
        second_per_grids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            use_audio_in_video (`bool`, *optional*):
                 If set to `True`, use the audio in video.
            audio_seqlens (`torch.LongTensor` of shape `(num_audios)`, *optional*):
                The length of feature shape of each audio in LLM.
            second_per_grids (`torch.LongTensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.spatial_merge_size
        image_token_id = self.config.image_token_index
        video_token_id = self.config.video_token_index
        audio_token_id = self.config.audio_token_index
        vision_start_token_id = self.config.vision_start_token_id
        audio_start_token_id = self.config.audio_start_token_id
        position_id_per_seconds = self.config.position_id_per_seconds
        seconds_per_chunk = self.config.seconds_per_chunk

        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_idx, video_idx, audio_idx = 0, 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums, audio_nums = 0, 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                audio_nums = torch.sum(input_ids == audio_start_token_id)
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (
                    (vision_tokens == audio_start_token_id).sum()
                    if use_audio_in_video
                    else (vision_tokens == video_token_id).sum()
                )
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos, remain_audios = image_nums, video_nums, audio_nums
                multimodal_nums = (
                    image_nums + audio_nums if use_audio_in_video else image_nums + video_nums + audio_nums
                )
                for _ in range(multimodal_nums):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if audio_token_id in input_tokens and remain_audios > 0:
                        ed_audio = input_tokens.index(audio_token_id, st)
                    else:
                        ed_audio = len(input_tokens) + 1
                    min_ed = min(ed_image, ed_video, ed_audio)
                    if min_ed == ed_audio:
                        text_len = min_ed - st - 1
                        if text_len != 0:
                            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        bos_len = 1
                        llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        audio_len = ((audio_seqlens[audio_idx] - 1) // 2 + 1 - 2) // 2 + 1
                        llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                        llm_pos_ids_list.append(llm_pos_ids)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        eos_len = 1
                        llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                        st += text_len + bos_len + audio_len + eos_len
                        audio_idx += 1
                        remain_audios -= 1

                    elif min_ed == ed_image:
                        text_len = min_ed - st - 1
                        if text_len != 0:
                            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        bos_len = 1
                        llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        grid_t = image_grid_thw[image_idx][0]
                        grid_hs = image_grid_thw[:, 1]
                        grid_ws = image_grid_thw[:, 2]
                        t_index = (torch.arange(grid_t) * 1 * position_id_per_seconds).long()
                        llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )
                        image_len = image_grid_thw[image_idx].prod() // (spatial_merge_size**2)
                        llm_pos_ids_list.append(llm_pos_ids)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        eos_len = 1
                        llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                        st += text_len + bos_len + image_len + eos_len
                        image_idx += 1
                        remain_images -= 1

                    elif min_ed == ed_video and not use_audio_in_video:
                        text_len = min_ed - st - 1
                        if text_len != 0:
                            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        bos_len = 1
                        llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        grid_t = video_grid_thw[video_idx][0]
                        grid_hs = video_grid_thw[:, 1]
                        grid_ws = video_grid_thw[:, 2]
                        t_index = (
                            torch.arange(grid_t) * second_per_grids[video_idx].cpu().float() * position_id_per_seconds
                        ).long()
                        llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )
                        video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)
                        llm_pos_ids_list.append(llm_pos_ids)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        eos_len = 1
                        llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                        st += text_len + bos_len + video_len + eos_len
                        video_idx += 1
                        remain_videos -= 1

                    elif min_ed == ed_video and use_audio_in_video:
                        text_len = min_ed - st - 2
                        if text_len != 0:
                            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        bos_len = 1
                        llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)
                        llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        audio_len = ((audio_seqlens[audio_idx] - 1) // 2 + 1 - 2) // 2 + 1
                        audio_llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                        grid_t = video_grid_thw[video_idx][0]
                        grid_hs = video_grid_thw[:, 1]
                        grid_ws = video_grid_thw[:, 2]

                        t_index = (
                            torch.arange(grid_t) * second_per_grids[video_idx].cpu().float() * position_id_per_seconds
                        ).long()
                        video_llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )

                        t_ntoken_per_chunk = int(position_id_per_seconds * seconds_per_chunk)
                        video_chunk_indexes = self.get_chunked_index(video_llm_pos_ids, t_ntoken_per_chunk, st_idx)
                        audio_chunk_indexes = self.get_chunked_index(audio_llm_pos_ids, t_ntoken_per_chunk, st_idx)
                        sub_len = 0
                        for j in range(max(len(video_chunk_indexes), len(audio_chunk_indexes))):
                            video_chunk_index = video_chunk_indexes[j] if j < len(video_chunk_indexes) else None
                            audio_chunk_index = audio_chunk_indexes[j] if j < len(audio_chunk_indexes) else None
                            if video_chunk_index is not None:
                                sub_len += video_chunk_index[1] - video_chunk_index[0]

                                llm_pos_ids_list.append(
                                    video_llm_pos_ids[:, video_chunk_index[0] : video_chunk_index[1]]
                                )
                            if audio_chunk_index is not None:
                                sub_len += audio_chunk_index[1] - audio_chunk_index[0]

                                llm_pos_ids_list.append(
                                    audio_llm_pos_ids[:, audio_chunk_index[0] : audio_chunk_index[1]]
                                )
                        video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        eos_len = 1
                        llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)
                        llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                        st += text_len + bos_len * 2 + audio_len + video_len + eos_len * 2

                        audio_idx += 1
                        video_idx += 1
                        remain_videos -= 1
                        remain_audios -= 1

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)

                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)

            return position_ids, mrope_position_deltas
        else:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)

            return position_ids, mrope_position_deltas


############################
#      Start Thinker       #
############################


@dataclass
class Qwen2_5OmniThinkerCausalLMOutputWithPast(ModelOutput):

    #是一个 数据容器类 ，专门用来封装模型（Thinker/LLM部分）的输出结果。
    """
    Base class for Qwen2.5OmniThinker causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`, *optional*):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None   # rope_deltas (RoPE 偏移量 - Omni 特有) :




    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor = None
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q = self.q(hidden_states).reshape(seq_length, self.num_heads, -1)
        k = self.k(hidden_states).reshape(seq_length, self.num_heads, -1)
        v = self.v(hidden_states).reshape(seq_length, self.num_heads, -1)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen2_5OmniMLP(nn.Module):
    def __init__(self, config, bias: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class Qwen2_5OmniAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Qwen2_5OmniConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.rope_scaling = config.rope_scaling

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = None
        # self.rotary_emb = Qwen2_5OmniRotaryEmbedding(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Fix precision issues in Qwen2-VL float16 inference
        # Replace inf values with zeros in attention weights to prevent NaN propagation
        if query_states.dtype == torch.float16:
            attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen2MLP(nn.Module):
    def __init__(self, config, bias: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class Qwen2_5OmniSdpaAttention(Qwen2_5OmniAttention):
    """
    Qwen2 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Qwen2Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from Qwen2Attention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Qwen2_5OmniModel is using Qwen2_5OmniSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


QWEN2_5_OMNI_ATTENTION_CLASSES = {
    "eager": Qwen2_5OmniAttention,
    # "flash_attention_2": Qwen2_5OmniFlashAttention2,
    "sdpa": Qwen2_5OmniSdpaAttention,
}


class Qwen2_5OmniDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2_5OmniConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = QWEN2_5_OMNI_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


QWEN2_5OMNI_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`{config_class}`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Qwen2.5OmniThinker Model outputting raw hidden-states without any specific head on top.",
    QWEN2_5OMNI_START_DOCSTRING.format(config_class="Qwen2_5OmniTextConfig"),
)
class Qwen2_5OmniThinkerTextModel(Qwen2_5OmniPreTrainedModel):
    config_class = Qwen2_5OmniTextConfig
    _no_split_modules = ["Qwen2_5OmniDecoderLayer"]

    def __init__(self, config: Qwen2_5OmniTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2_5OmniDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.rotary_emb = Qwen2_5OmniRotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: Union[torch.Tensor, "BlockMask"],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Qwen25OmniThinkerText. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            return attention_mask

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Qwen2_5OmniConfig,
        past_key_values: Cache,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to place the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`Qwen25OmniThinkerTextConfig`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            if config.get_text_config().sliding_window is not None:
                # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
                # the check is needed to verify is current checkpoint was trained with sliding window or not
                if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=device) <= (
                        cache_position.reshape(-1, 1) - config.get_text_config().sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask


QWEN2_5OMNITHINKER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        input_features (`torch.FloatTensor` of shape `(batch_size, feature_size, feature_sequence_length)`):
            Float values mel features extracted from the raw speech waveform. Raw speech waveform can be obtained by
            loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via
            the soundfile library (`pip install soundfile`). To prepare the array into `input_features`, the
            [`AutoFeatureExtractor`] should be used for extracting the mel features, padding and conversion into a
            tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size), *optional*):
            The tensors corresponding to the input images. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`SiglipImageProcessor.__call__`] for details ([]`NewTaskModelProcessor`] uses
            [`SiglipImageProcessor`] for processing images).
        pixel_values_videos(`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size), *optional*):
            The tensors corresponding to the input videos. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`SiglipImageProcessor.__call__`] for details ([]`NewTaskModelProcessor`] uses
            [`SiglipImageProcessor`] for processing videos).
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        feature_attention_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`, *optional*):
            Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        audio_feature_lengths (`torch.LongTensor` of shape `(num_audios)`, *optional*):
            The length of feature shape of each audio in LLM.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        past_key_values (`list(torch.FloatTensor)`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    """The Qwen2.5OmniThinker model which consists of a audio backbone and a language model.""",
    QWEN2_5OMNI_START_DOCSTRING.format(config_class="Qwen2_5OmniThinkerConfig"),
)

# thinker
# Qwen2_5OmniThinkerForConditionalGeneration，处理文本、音频、图像、视频输入并生成文本 token 序列
class Qwen2_5OmniThinkerForConditionalGeneration(Qwen2_5OmniPreTrainedModelForConditionalGeneration, GenerationMixin):
    config_class = Qwen2_5OmniThinkerConfig
    base_model_prefix = "thinker"
    _no_split_modules = ["Qwen2_5OmniAudioEncoder", "Qwen2_5OmniVisionEncoder"]

    def __init__(self, config: Qwen2_5OmniThinkerConfig):
        super().__init__(config)
        print("========",config._attn_implementation)
        ##############
        self.audio_tower = None
        # self.audio_tower = Qwen2_5OmniAudioEncoder._from_config(
        #     config.audio_config, attn_implementation=config._attn_implementation
        # )
################# 注释掉audio
        self.visual = None
        # self.visual = Qwen2_5OmniVisionEncoder._from_config(
        #     config.vision_config, attn_implementation=config._attn_implementation
        # )
        # EEG embeddings are provided externally; do not instantiate an encoder here.
        self.eeg_encoder = None
        # self.audio_tower.cpu()
        # self.visual.cpu()

        self.vocab_size = config.text_config.vocab_size
        self.model = Qwen2_5OmniThinkerTextModel._from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.spatial_merge_size = config.vision_config.spatial_merge_size
        self.rope_deltas = None
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    @add_start_docstrings_to_model_forward(QWEN2_5OMNITHINKER_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=Qwen2_5OmniThinkerCausalLMOutputWithPast, config_class="Qwen2_5OmniThinkerConfig"
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        ##########
        eeg_embeds_text: Optional[torch.FloatTensor] = None,
        eeg_embeds_image: Optional[torch.FloatTensor] = None,
        ##########
        eeg_embeds: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_audio_in_video: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        video_second_per_grid: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, Qwen2_5OmniThinkerCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from io import BytesIO
        >>> from urllib.request import urlopen
        >>> import librosa
        >>> from qwen_vl_utils import process_vision_info
        >>> from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

        >>> thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-7B")
        >>> processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

        >>> conversations = [
        >>>         {'role': 'system', 'content': 'You are a helpful voice chat bot, and please respond to me in a casual conversation manner using random voice.'},
        >>>         {"role": "user", "content": [
        >>>             {"type": "image", "image_url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
        >>>             {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"},
        >>>         ]},
        >>> ]

        >>> text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        >>> audios = [ librosa.load(BytesIO(urlopen( conversations[1]['content'][1]['audio_url'] ).read()), sr=self.processor.feature_extractor.sampling_rate) ]
        >>> images, videos = process_vision_info(conversations)
        >>> inputs = processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True)

        >>> # Generate
        >>> inputs['use_audio_in_video'] = `True` or `False`
        >>> generation = thinker.generate(**inputs, max_new_tokens=2048)
        >>> generate_ids = generation[:, inputs.input_ids.size(1):]

        >>> response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        ```"""
        if self.visual is not None:
            self.visual.to('cpu')
        if self.audio_tower is not None:
            self.audio_tower.to('cpu')
        torch.cuda.empty_cache()

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
        else:
            audio_feature_lengths = None
        if attention_mask is not None and position_ids is None:
            if (
                cache_position is None
                or (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
            ):
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                    use_audio_in_video,
                    audio_feature_lengths,
                    video_second_per_grid,
                )
                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = input_ids.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if inputs_embeds is None:
            # 1. Extract the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # 2. Merge text , audios , image and video
        if input_ids is not None and input_ids.shape[1] != 1:  # Prefill stage
            if input_features is not None and self.audio_tower is not None:
                self.audio_tower.to('cuda')
                audio_feat_lengths, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(
                    audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
                )
                feature_lens = (
                    audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
                )
                audio_outputs = self.audio_tower(
                    input_features,
                    feature_lens=feature_lens,
                    aftercnn_lens=audio_feat_lengths,
                )
                audio_features = audio_outputs.last_hidden_state
                if audio_features.shape[0] != sum(audio_output_lengths.tolist()):
                    raise ValueError("length of audio_features should match audio_output_lengths")
                audio_mask = (
                    (input_ids == self.config.audio_token_index)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)
                
                if self.audio_tower is not None:
                    self.audio_tower.to('cpu')
                torch.cuda.empty_cache()
#####################
            if eeg_embeds_text is not None and hasattr(self.config, "eeg_text_token_index"):
                num_text_tokens = (input_ids == self.config.eeg_text_token_index).sum(dim=1)
                if not torch.all(num_text_tokens == 1):
                    raise ValueError("EEG text embedding requires exactly one EEG text token per sample")
                text_mask = (
                    (input_ids == self.config.eeg_text_token_index)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                eeg_embeds_text = eeg_embeds_text.to(inputs_embeds.device, inputs_embeds.dtype).unsqueeze(1)
                inputs_embeds = inputs_embeds.masked_scatter(text_mask, eeg_embeds_text)

            if eeg_embeds_image is not None and hasattr(self.config, "eeg_image_token_index"):
                num_image_tokens = (input_ids == self.config.eeg_image_token_index).sum(dim=1)
                if not torch.all(num_image_tokens == 1):
                    raise ValueError("EEG image embedding requires exactly one EEG image token per sample")
                image_mask_eeg = (
                    (input_ids == self.config.eeg_image_token_index)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                eeg_embeds_image = eeg_embeds_image.to(inputs_embeds.device, inputs_embeds.dtype).unsqueeze(1)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask_eeg, eeg_embeds_image)

            if eeg_embeds is not None and hasattr(self.config, "eeg_token_index"):
                num_eeg_tokens = (input_ids == self.config.eeg_token_index).sum(dim=1)
                if not torch.all(num_eeg_tokens == 1):
                    raise ValueError("EEG embedding requires exactly one EEG token per sample")
                eeg_mask = (
                    (input_ids == self.config.eeg_token_index)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                eeg_embeds = eeg_embeds.to(inputs_embeds.device, inputs_embeds.dtype).unsqueeze(1)
                inputs_embeds = inputs_embeds.masked_scatter(eeg_mask, eeg_embeds)
###############
            if pixel_values is not None and self.visual is not None:
                self.visual.to('cuda')
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = (
                    (input_ids == self.config.image_token_index)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                self.visual.to('cpu')
                torch.cuda.empty_cache()
                

            if pixel_values_videos is not None and self.visual is not None:
                self.visual.to('cuda')
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                video_mask = (
                    (input_ids == self.config.video_token_index)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
                self.visual.to('cpu')
                torch.cuda.empty_cache()
                

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)

        if not return_dict:
            output = (logits,) + outputs
            return (loss,) + output if loss is not None else output

        return Qwen2_5OmniThinkerCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        input_features=None,
        feature_attention_mask=None,
        use_audio_in_video=False,
        video_second_per_grid=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            use_audio_in_video=use_audio_in_video,
            video_second_per_grid=video_second_per_grid,
            **kwargs,
        )

        model_inputs["position_ids"] = None

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs


############################
#       Start Talker       #
############################


@add_start_docstrings(
    "The bare Qwen2.5OmniTalker Model outputting raw hidden-states without any specific head on top.",
    QWEN2_5OMNI_START_DOCSTRING.format(config_class="Qwen2_5OmniTalkerConfig"),
)

############################
#      Start Token2Wav     #
############################





@add_start_docstrings(
    "The full Qwen2.5Omni Token2WavBigVGAN model. Which take mel spectrogram as input and predict waveform.",
    QWEN2_5OMNI_START_DOCSTRING.format(config_class="Qwen2_5OmniBigVGANConfig"),
)

@add_start_docstrings(
    "The full Qwen2.5Omni Token2WavDiT model. Which take speech tokens as input and predict mel spectrogram.",
    QWEN2_5OMNI_START_DOCSTRING.format(config_class="Qwen2_5OmniDiTConfig"),
)
@add_start_docstrings(
    "The full Qwen2.5Omni Token2Wav model. Consists a DiT model take speech tokens as input and predict mel spectrogram and a BigVGAN vocoder take mel spectrogram as input and predict waveform.",
    QWEN2_5OMNI_START_DOCSTRING.format(config_class="Qwen2_5OmniToken2WavConfig"),
)
############################
#    Start Qwen2.5Omni     #
############################


@add_start_docstrings(
    """
    The full Qwen2.5Omni model, a multimodal model composed of 3 sub-models:
    - [`Qwen2_5OmniThinkerForConditionalGeneration`]:
    a causal auto-regressive transformer takes text, audio, image, video as input and predict text tokens.
    - [`Qwen2_5OmniTalkerForConditionalGeneration`]:
    a causal auto-regressive transformer takes thinker hidden states and response as input and predict speech tokens.
    - [`Qwen2_5OmniToken2WavModel`]:
    a DiT model take speech tokens as input and predict mel spectrogram and a BigVGAN vocoder take mel spectrogram as input and predict waveform.
    """,
    QWEN2_5OMNI_START_DOCSTRING.format(config_class=Qwen2_5OmniConfig),
)

# 下游产出
# Qwen2_5OmniForConditionalGeneration 统一调度各子模型，负责多模态输入与文本/语音的联合生成
class Qwen2_5OmniForConditionalGeneration(Qwen2_5OmniPreTrainedModel, GenerationMixin):
    config_class = Qwen2_5OmniConfig
    _no_split_modules = [
        "Qwen2_5OmniTalkerForConditionalGeneration",
        "Qwen2_5OmniToken2WavModel",
    ]

    def __init__(self, config):
        super().__init__(config)

        self.thinker = Qwen2_5OmniThinkerForConditionalGeneration(config.thinker_config)

        self.has_talker = False # config.enable_audio_output
        self.speaker_map = {}
        # if config.enable_audio_output:
        #     self.enable_talker()

        # Initialize EEG components placeholders
        self.eeg_encoder = None
        self.qformer = None
        
        # Linear layer to project 512 (CLIP dim) -> Thinker Hidden Size
        thinker_hidden_size = config.thinker_config.text_config.hidden_size
        self.eeg_projector = nn.Linear(512, thinker_hidden_size)
        
    # def enable_talker(self):
    #     self.talker = Qwen2_5OmniTalkerForConditionalGeneration(self.config.talker_config)
    #     self.token2wav = Qwen2_5OmniToken2WavModel(self.config.token2wav_config)
    #     # self.token2wav.float()
    #     self.token2wav.half()
    #     # self.token2wav.cpu()
        
        self.has_talker = False

    def load_speakers(self, path):
        for key, value in torch.load(path).items():
            self.speaker_map[key] = value
        logger.info("Speaker {} loaded".format(list(self.speaker_map.keys())))

    def disable_talker(self):
        if hasattr(self, "talker"):
            del self.talker
        if hasattr(self, "token2wav"):
            del self.token2wav
        self.has_talker = False

    def get_input_embeddings(self):
        return self.thinker.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.thinker.set_input_embeddings(value)
#####
    def generate_from_eeg(
        self,
        eeg_input: torch.Tensor,
        tokenizer: Optional[object] = None,
        prompt_text: str = "Describe the image decoded from brain signals.",
        eeg_token: str = "<EEG>",
        max_new_tokens: int = 128,
        **kwargs,
    ):
        if self.eeg_encoder is None:
            raise ValueError("EEG components are not initialized.")
        emb_img, emb_txt, _ = self.eeg_encoder(eeg_input)
        if getattr(self, "use_qformer", False) and self.qformer is not None:
            eeg_embed = self.qformer(torch.stack([emb_img, emb_txt], dim=1), return_sequence=False)
        else:
            eeg_embed = self.eeg_projector(emb_txt)
        if tokenizer is not None:
            if eeg_token not in tokenizer.get_vocab():
                tokenizer.add_tokens([eeg_token])
                self.thinker.resize_token_embeddings(len(tokenizer))
            eeg_tok_id = tokenizer.convert_tokens_to_ids(eeg_token)
            self.thinker.config.eeg_token_index = eeg_tok_id
            enc = tokenizer(prompt_text, return_tensors="pt")
            input_ids = torch.cat([
                torch.tensor([[eeg_tok_id]], dtype=torch.long, device=self.thinker.device),
                enc["input_ids"].to(self.thinker.device),
            ], dim=1)
            generation = self.thinker.generate(
                input_ids=input_ids,
                eeg_embeds=eeg_embed.to(self.thinker.device),
                max_new_tokens=max_new_tokens,
                **kwargs,
            )
            gen_ids = generation[:, input_ids.size(1):]
            return gen_ids
        if "input_ids" in kwargs:
            input_ids = kwargs["input_ids"].to(self.thinker.device)
            if "eeg_token_index" in kwargs:
                self.thinker.config.eeg_token_index = kwargs["eeg_token_index"]
            generation = self.thinker.generate(
                input_ids=input_ids,
                eeg_embeds=eeg_embed.to(self.thinker.device),
                max_new_tokens=max_new_tokens,
                **kwargs,
            )
            return generation
        raise ValueError("Provide a tokenizer or prebuilt input_ids to generate from EEG.")

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
############
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return self.thinker(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        config=None,
        cache_dir=None,
        ignore_mismatched_sizes=False,
        force_download=False,
        local_files_only=False,
        token=None,
        revision="main",
        use_safetensors=None,
        weights_only=True,
        **kwargs,
    ):
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            weights_only=weights_only,
            **kwargs,
        )
        spk_path = cached_file(
            pretrained_model_name_or_path,
            "spk_dict.pt",
            subfolder=kwargs.pop("subfolder", None),
            cache_dir=kwargs.pop("cache_dir", None),
            force_download=kwargs.pop("force_download", False),
            proxies=kwargs.pop("proxies", None),
            resume_download=kwargs.pop("resume_download", None),
            local_files_only=kwargs.pop("local_files_only", False),
            token=kwargs.pop("use_auth_token", None),
            revision=kwargs.pop("revision", None),
        )
        if spk_path is None:
            raise ValueError(f"""{pretrained_model_name_or_path}/{spk_path} not exists""")
        model.load_speakers(spk_path)

        return model

    @torch.no_grad()
    # TODO: raushan, defaults should be saved in generation config
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        speaker: str = "Chelsie",
        use_audio_in_video: bool = False,
        return_audio: Optional[bool] = None,
        thinker_max_new_tokens: int = 1024,
        talker_max_new_tokens: int = 4096,
        talker_do_sample: bool = True,
        talker_top_k: int = 40,
        talker_top_p: float = 0.8,
        talker_temperature: float = 0.9,
        talker_eos_token_id: list[int] = [8292, 8294],
        talker_repetition_penalty: float = 1.05,
        **kwargs,
    ):
        r"""
        Generate text response and audio from input.

        Args:
            input_ids (`Optional[torch.Tensor]`, *optional*):
                Input ids, should obtain from processor.
            speaker (`str` , defaults to "Chelsie"):
                Which speaker should be used in audio response.
            use_audio_in_video (`bool`, defaults to False):
                Whether or not use audio track in video, should same as the parameter in `process_audio_info`.
            return_audio (`Optional[bool]`, *optional*):
                Whether or not return response in audio format. When `return_audio=None`, this parameter is same as `config.enable_audio_output`.
            kwargs (*optional*):
                - Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model.
                - With a *thinker_*, *talker_*, *token2wav_* prefix, they will be input for the `generate` method of the
                thinker, talker and token2wav respectively. It has the priority over the keywords without a prefix.
        Returns:
            When `return_audio=False`:
                - **Text** (`torch.Tensor`): Generated text token sequence.
            When `return_audio=True`:
                - **Text** (`torch.Tensor`): Generated text token sequence.
                - **Audio waveform** (`torch.Tensor`): Generated audio waveform.
        """
        self.talker.to("cpu")
        self.token2wav.to("cpu")
        torch.cuda.empty_cache()
        self.thinker.to("cuda")
        
        if speaker not in self.speaker_map:
            raise ValueError(f"{speaker} is not availible, availible speakers: {self.speaker_map.keys()}")
        if return_audio and not self.has_talker:
            raise ValueError(
                "Cannot use talker when talker module not initalized. Use `enable_talker` method or set enable_talker in config to enable talker."
            )
        if return_audio is None:
            return_audio = self.has_talker
        if input_ids.shape[0] != 1 and return_audio:
            raise NotImplementedError("Qwen2.5-Omni currently does not support batched inference with audio output")

        shared_kwargs = {"use_audio_in_video": use_audio_in_video}
        thinker_kwargs = {
            "max_new_tokens": thinker_max_new_tokens,
        }
        talker_kwargs = {
            "max_new_tokens": talker_max_new_tokens,
            "do_sample": talker_do_sample,
            "top_k": talker_top_k,
            "top_p": talker_top_p,
            "temperature": talker_temperature,
            "eos_token_id": talker_eos_token_id,
            "repetition_penalty": talker_repetition_penalty,
        }
        token2wav_kwargs = {}

        for key, value in kwargs.items():
            if key.startswith("thinker_"):
                thinker_kwargs[key[len("thinker_") :]] = value
            elif key.startswith("talker_"):
                talker_kwargs[key[len("talker_") :]] = value
            elif key.startswith("token2wav_"):
                token2wav_kwargs[key[len("token2wav_") :]] = value
            # Process special input values
            elif key == "feature_attention_mask":
                thinker_kwargs[key] = value
                talker_kwargs["audio_feature_lengths"] = torch.sum(value, dim=1)
            elif key == "input_features" or key == "attention_mask":
                thinker_kwargs[key] = value
            # Put other key to shared kwargs
            else:
                shared_kwargs[key] = value

        # Merge kwargs
        for key, value in shared_kwargs.items():
            if key not in thinker_kwargs:
                thinker_kwargs[key] = value
            if key not in talker_kwargs:
                talker_kwargs[key] = value
            if key not in token2wav_kwargs:
                token2wav_kwargs[key] = value
        speaker_params = self.speaker_map[speaker]

        # 1. Generate from thinker module
        generate_audio = return_audio and self.has_talker
        if generate_audio:
            thinker_kwargs["output_hidden_states"] = True
            thinker_kwargs["return_dict_in_generate"] = True

        thinker_result = self.thinker.generate(input_ids=input_ids, **thinker_kwargs)

        if not generate_audio:
            return thinker_result
        
        thinker_embed_tokens = self.thinker.get_input_embeddings()

        self.thinker.to("cpu")
        thinker_embed_tokens.to("cuda")
        torch.cuda.empty_cache()
        
        self.talker.to("cuda")


        # 2. Generate speech tokens from talker module
        thinker_generate_ids = thinker_result.sequences[:, input_ids.size(1) :].to(self.talker.device)
        thinker_token_embeds = [
            token_hidden_states[0].to(self.talker.device) for token_hidden_states in thinker_result.hidden_states
        ]
        thinker_hidden_states = [
            token_hidden_states[-1].to(self.talker.device) for token_hidden_states in thinker_result.hidden_states
        ]

        talker_text_bos_token = speaker_params["bos_token"]
        talker_input_text_ids = torch.cat(
            [
                input_ids.to(self.talker.device),
                torch.tensor([[talker_text_bos_token]], dtype=torch.long, device=self.talker.device),
                thinker_generate_ids[:, :1],
            ],
            dim=-1,
        )

        talker_input_ids = torch.cat(
            [
                torch.full_like(input_ids, fill_value=self.talker.codec_mask_token, device=self.talker.device),
                torch.tensor([[self.talker.codec_pad_token]], dtype=torch.long, device=self.talker.device),
                torch.tensor([[self.talker.codec_bos_token]], dtype=torch.long, device=self.talker.device),
            ],
            dim=1,
        )

        # thinker_embed_tokens = self.thinker.get_input_embeddings()
        thinker_reply_part = torch.cat(thinker_hidden_states[1:], dim=1) + torch.cat(thinker_token_embeds[1:], dim=1)
        talker_inputs_embeds = thinker_hidden_states[0] + thinker_token_embeds[0]
        talker_text_bos_token = torch.tensor([[talker_text_bos_token]], dtype=torch.long, device=self.talker.device)
        talker_text_bos_embed = thinker_embed_tokens(talker_text_bos_token).to(self.talker.device)
        talker_inputs_embeds = torch.cat(
            [
                talker_inputs_embeds,
                talker_text_bos_embed,
                thinker_reply_part[:, :1, :],
            ],
            dim=1,
        )

        eos_embedding = thinker_embed_tokens(
            torch.tensor([[self.talker.text_eos_token]], dtype=torch.long, device=self.talker.device)
        ).to(self.talker.device)

        pad_embedding = thinker_embed_tokens(
            torch.tensor([[self.talker.text_pad_token]], dtype=torch.long, device=self.talker.device)
        ).to(self.talker.device)

        thinker_reply_part = torch.cat(
            [
                thinker_reply_part[:, 1:, :],
                eos_embedding,
                pad_embedding,
            ],
            dim=1,
        )

        talker_attention_mask = None
        if "attention_mask" in kwargs:
            talker_attention_mask = torch.cat(
                [kwargs["attention_mask"], kwargs["attention_mask"].new_ones((1, 2))], dim=1
            ).to(self.talker.device)

        talker_result = self.talker.generate(
            input_ids=talker_input_ids,
            input_text_ids=talker_input_text_ids,
            thinker_reply_part=thinker_reply_part,
            inputs_embeds=talker_inputs_embeds,
            attention_mask=talker_attention_mask,
            suppress_tokens=[self.talker.codec_bos_token],
            **{k: (v.to(self.talker.device) if torch.is_tensor(v) else v) for k, v in talker_kwargs.items()},
        )
        talker_generate_codes = talker_result[:, talker_input_ids.shape[1] : -1]
        self.talker.to("cpu")
        torch.cuda.empty_cache()

        # 3. Generate wavs from code
        self.token2wav.to("cuda")

        wav = self.token2wav(
            talker_generate_codes.to(self.token2wav.device),
            conditioning=speaker_params["cond"].to(self.token2wav.device).half(),
            reference_mel=speaker_params["ref_mel"].to(self.token2wav.device).half(),
            return_audio_in_chunk=True,
            **token2wav_kwargs,
        )
        torch.cuda.empty_cache()
        self.thinker.to("cuda")
        
        return thinker_result.sequences, talker_generate_codes, wav.float()


__all__ = [
    "Qwen2_5OmniForConditionalGeneration",
    "Qwen2_5OmniThinkerTextModel",
    "Qwen2_5OmniThinkerForConditionalGeneration",
    "Qwen2_5OmniTalkerModel",
    "Qwen2_5OmniTalkerForConditionalGeneration",
    "Qwen2_5OmniToken2WavDiTModel",
    "Qwen2_5OmniToken2WavBigVGANModel",
    "Qwen2_5OmniToken2WavModel",
    "Qwen2_5OmniPreTrainedModel",
    "Qwen2_5OmniPreTrainedModelForConditionalGeneration",
]
