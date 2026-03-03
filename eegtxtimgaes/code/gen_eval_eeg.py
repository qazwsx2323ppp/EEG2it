import os, sys
import json
import numpy as np
import torch
from einops import rearrange
from PIL import Image
import torchvision.transforms as transforms
from config import *
import wandb
import datetime
import argparse


from config import Config_Generative_Model
from dataset import create_EEG_dataset
from dc_ldm.ldm_for_eeg import eLDM_eval
from eval_metrics import get_similarity_metric

def to_image(img):
    if img.shape[-1] != 3:
        img = rearrange(img, 'c h w -> h w c')
    img = 255. * img
    return Image.fromarray(img.astype(np.uint8))

def channel_last(img):
    if img.shape[-1] == 3:
        return img
    return rearrange(img, 'c h w -> h w c')

def normalize(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = img * 2.0 - 1.0 # to -1 ~ 1
    return img

def wandb_init(config):
    wandb.init( project="dreamdiffusion",
                group='eval',
                anonymous="allow",
                config=config,
                reinit=True)

class random_crop:
    def __init__(self, size, p):
        self.size = size
        self.p = p
    def __call__(self, img):
        if torch.rand(1) < self.p:
            return transforms.RandomCrop(size=(self.size, self.size))(img)
        return img

def get_eval_metric(samples, avg=True):
    """
    复制自 eeg_ldm.py，用于对生成结果计算 MSE/PCC/SSIM 以及 top-1-class 指标。
    samples: numpy array, shape [N, B, C, H, W]，第 0 张为 GT，其余为生成结果。
    """
    metric_list = ['mse', 'pcc', 'ssim', 'psm']
    res_list = []
    
    gt_images = [img[0] for img in samples]
    gt_images = rearrange(np.stack(gt_images), 'n c h w -> n h w c')
    samples_to_run = np.arange(1, len(samples[0])) if avg else [1]
    for m in metric_list:
        res_part = []
        for s in samples_to_run:
            pred_images = [img[s] for img in samples]
            pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
            res = get_similarity_metric(pred_images, gt_images, method='pair-wise', metric_name=m)
            res_part.append(np.mean(res))
        res_list.append(np.mean(res_part))     
    res_part = []
    for s in samples_to_run:
        pred_images = [img[s] for img in samples]
        pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
        res = get_similarity_metric(pred_images, gt_images, 'class', None, 
                        n_way=50, num_trials=50, top_k=1, device='cuda' if torch.cuda.is_available() else 'cpu')
        res_part.append(np.mean(res))
    res_list.append(np.mean(res_part))
    res_list.append(np.max(res_part))
    metric_list.append('top-1-class')
    metric_list.append('top-1-class (max)')
    return res_list, metric_list

def get_args_parser():
    parser = argparse.ArgumentParser('Double Conditioning LDM Finetuning', add_help=False)
    # project parameters
    parser.add_argument('--root', type=str, default='../dreamdiffusion/')
    parser.add_argument('--dataset', type=str, default='GOD')
    parser.add_argument('--model_path', type=str)

    parser.add_argument('--splits_path', type=str, default=None,
                        help='Path to dataset splits.')
    parser.add_argument('--eeg_signals_path', type=str, default=None,
                        help='Path to EEG signals data.')

    parser.add_argument('--config_patch', type=str, default=None,
                        help='sd config path.')
    
    parser.add_argument('--imagenet_path', type=str, default=None,
                        help='imagenet path.')

    parser.add_argument('--limit', type=int, default=None,
                        help='Max number of test samples to generate (default: all). Use e.g. 20 for quick test.')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of images to generate per EEG (default: from checkpoint config).')
    parser.add_argument('--ddim_steps', type=int, default=None,
                        help='DDIM/PLMS steps (default: from checkpoint config).')

    return parser


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    root = args.root
    target = args.dataset

    sd = torch.load(args.model_path, map_location='cpu',weights_only=False)
    config = sd['config']
    # update paths
    config.root_path = root
    if args.num_samples is not None:
        config.num_samples = args.num_samples
    if args.ddim_steps is not None:
        config.ddim_steps = args.ddim_steps

    output_path = os.path.join(config.root_path, 'results', 'eval',  
                    '%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    crop_pix = int(config.crop_ratio*config.img_size)
    img_transform_train = transforms.Compose([
        normalize,
        transforms.Resize((512, 512)),
        # random_crop(config.img_size-crop_pix, p=0.5),
        # transforms.Resize((256, 256)), 
        channel_last
    ])
    img_transform_test = transforms.Compose([
        normalize, transforms.Resize((512, 512)), 
        channel_last
    ])

    
    dataset_train, dataset_test = create_EEG_dataset(eeg_signals_path = args.eeg_signals_path, 
                splits_path = args.splits_path, imagenet_path=args.imagenet_path,
                image_transform=[img_transform_train, img_transform_test], subject = 4)
    num_voxels = dataset_test.dataset.data_len



    # create generateive model
    generative_model = eLDM_eval(args.config_patch, num_voxels,
                device=device, pretrain_root=config.pretrain_gm_path, logger=config.logger,
                ddim_steps=config.ddim_steps, global_pool=config.global_pool, use_time_cond=config.use_time_cond)
    # m, u = model.load_state_dict(pl_sd, strict=False)
    generative_model.model.load_state_dict(sd['model_state_dict'], strict=False)
    print('load ldm successfully')
    state = sd['state']
    os.makedirs(output_path, exist_ok=True)
    print('Output directory:', os.path.abspath(output_path))
    grid, _ = generative_model.generate(dataset_train, config.num_samples, 
                config.ddim_steps, config.HW, 10) # generate 10 instances
    grid_imgs = Image.fromarray(grid.astype(np.uint8))
    
    grid_imgs.save(os.path.join(output_path, f'./samples_train.png'))

    grid, samples = generative_model.generate(dataset_test, config.num_samples, 
                config.ddim_steps, config.HW, limit=args.limit, state=state, output_path = output_path)
    grid_imgs = Image.fromarray(grid.astype(np.uint8))

    # 保存测试集拼接图
    test_grid_path = os.path.join(output_path, f'./samples_test.png')
    grid_imgs.save(test_grid_path)

    # 计算评估指标
    print('\nComputing evaluation metrics on generated test samples...')
    metrics, metric_names = get_eval_metric(samples, avg=config.eval_avg)
    metrics_dict = {name: float(val) for name, val in zip(metric_names, metrics)}

    for name, val in metrics_dict.items():
        print(f'{name}: {val:.6f}')

    # 保存到 JSON
    metrics_path = os.path.join(output_path, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, ensure_ascii=False, indent=2)

    print('\nSaved:')
    print('  Train grid:', os.path.abspath(os.path.join(output_path, 'samples_train.png')))
    print('  Test grid :', os.path.abspath(test_grid_path))
    print('  Metrics   :', os.path.abspath(metrics_path))
    print('Done. Results saved to:', os.path.abspath(output_path))