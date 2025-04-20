"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
from datetime import time
import os
from pyexpat import model

import numpy as np
from regex import P
import torch as th
import torchvision.utils as vutils
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
import sys
sys.path.append('/home/zzx/projects/rrg-timsbc/zzx')
 
from LTSF_DDBM.diffusion.ddbm import dist_util, logger
from LTSF_DDBM.diffusion.ddbm.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from LTSF_DDBM.diffusion.ddbm.random_util import get_generator
from LTSF_DDBM.diffusion.ddbm.karras_diffusion import karras_sample, forward_sample

# from datasets import load_data
from LTSF_DDBM.data_provider.data_factory import data_provider

from pathlib import Path

from PIL import Image
def get_workdir(exp):
    workdir = f'./workdir/{exp}'
    return workdir

def main():
    args = create_argparser().parse_args()

    workdir = os.path.dirname(args.model_path)

    ## assume ema ckpt format: ema_{rate}_{steps}.pt
    split = args.model_path.split("_")
    step = int(split[-1].split(".")[0])
    sample_dir = Path(workdir)/f'sample_{step}/w={args.guidance}_churn={args.churn_step_ratio}'
    dist_util.setup_dist()
    if dist.get_rank() == 0:

        sample_dir.mkdir(parents=True, exist_ok=True)
    logger.configure(dir=workdir)


    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
    )
    # state_dict_keys = set(model.state_dict().keys())
    # print("keys that model needs: ",state_dict_keys)
    # ckpt_keys = th.load(args.model_path, map_location="cpu").keys()
    # print("keys that ckpt has: ",sorted(ckpt_keys))
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model = model.to(dist_util.dev())
    
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    

    all_images = []
    

    # all_dataloaders = load_data(
    #     data_dir=args.data_dir,
    #     dataset=args.dataset,
    #     batch_size=args.batch_size,
    #     image_size=args.image_size,
    #     include_test=True,
    #     seed=args.seed,
    #     num_workers=args.num_workers,
    # )
    # if args.split =='train':
    #     dataloader = all_dataloaders[1]
    # elif args.split == 'test':
    #     dataloader = all_dataloaders[2]
    # else:
    #     raise NotImplementedError
    _ , low_freq_data, high_freq_data = data_provider(args, flag=args.split)  # check
    args.num_samples = len(low_freq_data.dataset)

    
    for i, data in enumerate(low_freq_data):
        # print("data shape: ",data[0].shape)
        # print("data 0: ",data[0])
        # print("data 1: ",data[1])
        # print("data 2: ",data[2])

        conv_layer = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(3, 3), stride=1, padding=1).to(dist_util.dev()).double()
        x0_image = data[0].to(dist_util.dev())
        x0_image = x0_image.unsqueeze(1)
        x0_image = x0_image.double()
        x0_image_conv = conv_layer(x0_image)
        x0 = x0_image_conv.to(dist_util.dev()) 
        
        y0_image = data[1].to(dist_util.dev())
        y0_image = y0_image.unsqueeze(1)
        y0_image = y0_image.double()
        y0_image_conv = conv_layer(y0_image)  # Shape: [batch_size, channels, time_steps, feature_dim]
        y0 = y0_image_conv.to(dist_util.dev()) 
        
        # index = data[2].to(dist_util.dev())

        x0 = F.pad(x0, (0, 25))  
        y0 = F.pad(y0, (0, 25)) 
        # index = F.pad(index, (0, 10))   
        x0 = x0.to(dist_util.dev())
        y0 = y0.to(dist_util.dev())
        # index = index.to(dist_util.dev())
        print("x0 shape: ",x0.shape)
        print("y0 shape: ",y0.shape)
        # print("index shape: ",index.shape)
        model_kwargs = {'xT': y0}

        # reduce_channels = nn.Conv2d(32, 6, kernel_size=1)
        # reduce_channels = reduce_channels.to(dist_util.dev()).double()
        # y0_reduced = reduce_channels(y0)
        # y0_reduced = y0_reduced.to(dist_util.dev())
        # print("y0_reduced shape: ",y0_reduced.shape)

        sample, path, nfe = karras_sample(
            diffusion,
            model,
            y0,
            x0,
            steps=args.steps,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
            clip_denoised=args.clip_denoised,
            sampler=args.sampler,
            sigma_min=diffusion.sigma_min,
            sigma_max=diffusion.sigma_max,
            churn_step_ratio=args.churn_step_ratio,
            rho=args.rho,
            guidance=args.guidance
        )
        

        # sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        # sample = sample.permute(0, 2, 3, 1)
        sample = sample.detach().cpu().numpy()
        y0 = y0.detach().cpu().numpy()
        print("sample.shape:", sample.shape)
        sample = th.tensor(sample)  # 转回 tensor
        sample = sample.mean(dim=1, keepdim=True)  # 取平均值恢复单通道
        print("After: sample shape: ", sample.shape)   
        # sample = sample.contiguous()
        
        # gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        # dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        # if index is not None:
        #     gathered_index = [th.zeros_like(index) for _ in range(dist.get_world_size())]
        #     dist.all_gather(gathered_index, index)
        #     gathered_samples = th.cat(gathered_samples)
        #     gathered_index = th.cat(gathered_index)
        #     gathered_samples = gathered_samples[th.argsort(gathered_index)]
        # else:
        #     gathered_samples = th.cat(gathered_samples)

        # num_display = min(32, sample.shape[0])
        # if i == 0 and dist.get_rank() == 0:
        #     vutils.save_image(sample.permute(0,3,1,2)[:num_display].float(), f'{sample_dir}/sample_{i}.png', normalize=True,  nrow=int(np.sqrt(num_display)))
        #     if x0 is not None:
        #         vutils.save_image(x0_image[:num_display], f'{sample_dir}/x_{i}.png',nrow=int(np.sqrt(num_display)))
        #     vutils.save_image(y0_image[:num_display]/2+0.5, f'{sample_dir}/y_{i}.png',nrow=int(np.sqrt(num_display)))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        batch_idx = 0  # 只画 batch 内的第一个样本
        feature_idx = 0  # 选取第一个 feature 进行可视化

        context_len = x0_image.shape[2]  # 过去的时间步
        horizon_len = y0_image.shape[2]  # 未来的时间步
        label_len = 48

        # 取出 batch 内的某个样本，并从 3 通道数据恢复成单通道
        x0_plot = x0[batch_idx, 0, :, feature_idx].detach().cpu().numpy()
        y0_plot = y0[batch_idx, 0, :, feature_idx]
        sample_plot = sample[batch_idx, 0, :, feature_idx]

        # 构造时间轴
        context_time = np.arange(context_len)  # 过去的时间步
        horizon_time = np.arange(context_len , context_len + horizon_len )  # 未来的时间步

        # 画图
        print("start plot...")
        plt.figure(figsize=(10, 5))
        plt.plot(context_time, x0_plot, label="Context (Input)", color="blue")
        plt.plot(horizon_time, y0_plot, label="Ground Truth (Future)", color="green")
        plt.plot(horizon_time, sample_plot, label="Predicted (Future)", color="red", linestyle="dashed")

        plt.axvline(x=context_len - 1, color="black", linestyle="dotted")  # 分割过去和未来
        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        plt.legend()

        # 保存图像
        save_path = f"lstf_ddbm_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
            
            
        # all_images.append(gathered_samples.detach().cpu().numpy())
        
        
    logger.log(f"created {len(all_images) * args.batch_size * dist.get_world_size()} samples")
        

    arr = np.concatenate(all_images, axis=0)
    arr = arr[:args.num_samples]
    
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(sample_dir, f"samples_{shape_str}_nfe{nfe}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        data_dir="", ## only used in bridge
        dataset='',
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        sampler="heun",
        split='train',
        churn_step_ratio=0.,
        rho=7.0,
        steps=40,
        # model_path="/home/zzx/projects/rrg-timsbc/zzx/LTSF_DDBM/ckpt/diode_ema_0.9999_440000.pt",
        model_path="/home/zzx/projects/rrg-timsbc/zzx/LTSF_DDBM/diffusion/workdir/ett256_256d_2025-04-08_09-17-21/ema_0.9999_100000.pt",
        exp="",
        seed=42,
        ts="",
        upscale=False,
        num_workers=2,
        guidance=1.,
        # new args
        N=40,
        GEN_SAMPLER="heun",
        BS=16,
        NGPU=4,
        
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
