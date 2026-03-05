import torch
from torch import nn

class EegOmniPipeline:
    def __init__(self, thinker, processor, eeg_encoder, eeg_token="<EEG>", text_projector=None, image_projector=None):
        self.thinker = thinker
        self.processor = processor
        self.eeg_encoder = eeg_encoder
        self.tokenizer = processor.tokenizer
        self.hidden_size = thinker.config.text_config.hidden_size
        if eeg_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([eeg_token])
            self.thinker.resize_token_embeddings(len(self.tokenizer))
        self.eeg_token = eeg_token
        self.eeg_tok_id = self.tokenizer.convert_tokens_to_ids(eeg_token)
        self.thinker.config.eeg_token_index = self.eeg_tok_id
        self.text_projector = text_projector or nn.Linear(eeg_encoder.encoder.embedding_dim, self.hidden_size)
        self.image_projector = image_projector or nn.Identity()

    def encode_eeg(self, eeg_tensor):
        out_img, out_txt = self.eeg_encoder(eeg_tensor)
        return out_img, out_txt

    def generate_text_from_eeg(self, conversation, eeg_tensor, max_new_tokens=128, num_eeg_tokens=1):
        out_img, out_txt = self.encode_eeg(eeg_tensor)
        eeg_txt_for_llm = self.text_projector(out_txt)
        eeg_embeddings = eeg_txt_for_llm if num_eeg_tokens == 1 else eeg_txt_for_llm.expand(num_eeg_tokens, -1)
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=text, return_tensors="pt")
        eeg_ids = torch.tensor([[self.eeg_tok_id] * num_eeg_tokens], dtype=torch.long)
        input_ids = torch.cat([eeg_ids, inputs["input_ids"]], dim=1)
        gen = self.thinker.generate(input_ids=input_ids, eeg_embeddings=eeg_embeddings, eeg_token_index=self.eeg_tok_id, max_new_tokens=max_new_tokens)
        gen_ids = gen[:, input_ids.size(1):]
        return self.processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def generate_image_from_eeg(self, eeg_tensor, image_generator_callable):
        out_img, _ = self.encode_eeg(eeg_tensor)
        cond = self.image_projector(out_img)
        return image_generator_callable(cond)