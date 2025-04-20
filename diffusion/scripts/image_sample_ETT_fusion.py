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
from LTSF_DDBM.data_provider.data_factory import data_provider, data_provider_origin
from LTSF_DDBM.data_provider import Two_dimensional_FFT_FAMnet

from pathlib import Path

from PIL import Image
def get_workdir(exp):
    workdir = f'./workdir/{exp}'
    return workdir

def plot_time_series_batch(batch, filename, max_display=16, focus_f=0, focus_c=0, ground_truth=None):
    """
    batch: Tensor or ndarray of shape (B, C, T, F)
    ground_truth: same shape as batch
    focus_f: index of feature dimension to visualize
    focus_c: index of sample-frequency dimension to visualize
    """
    batch = batch.detach().cpu().numpy() if hasattr(batch, 'detach') else batch
    B, C, T, F = batch.shape
    num_display = min(max_display, B)

    if ground_truth is not None:
        ground_truth = ground_truth.detach().cpu().numpy() if hasattr(ground_truth, 'detach') else ground_truth
        assert ground_truth.shape == batch.shape, "ground_truth shape must match batch shape"

    cols = int(np.ceil(np.sqrt(num_display)))
    rows = int(np.ceil(num_display / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()

    for i in range(num_display):
        ax = axes[i]
        pred = batch[i, focus_c, :, focus_f]  # shape: (T,)
        ax.plot(pred, label='Prediction', color='blue')

        if ground_truth is not None:
            gt = ground_truth[i, focus_c, :, focus_f]
            ax.plot(gt, '--', label='Ground Truth', color='orange')

        ax.set_title(f"Sample {i}")
        ax.grid(True)
        ax.legend(fontsize='x-small', loc='best')

    # Remove unused subplots
    for j in range(num_display, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()

def main():
    args = create_argparser().parse_args()

    workdir = os.path.dirname(args.low_model_path)

    ## assume ema ckpt format: ema_{rate}_{steps}.pt
    split = args.low_model_path.split("_")
    step = int(split[-1].split(".")[0])
    sample_dir = Path(workdir)/f'sample_{step}/w={args.guidance}_churn={args.churn_step_ratio}'
    dist_util.setup_dist()
    if dist.get_rank() == 0:

        sample_dir.mkdir(parents=True, exist_ok=True)
    logger.configure(dir=workdir)


    logger.log("creating low_freq model and diffusion...")
    low_model, low_diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
    )
    low_model.load_state_dict(dist_util.load_state_dict(args.low_model_path, map_location="cpu"))
    low_model.to(dist_util.dev()).eval()

    logger.log("creating high_freq model and diffusion...")
    high_model, high_diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
    )
    high_model.load_state_dict(dist_util.load_state_dict(args.high_model_path, map_location="cpu"))
    high_model.to(dist_util.dev()).eval()

    high_model_sample80, high_diffusion_sample80 = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
    )
    high_model_sample80.load_state_dict(dist_util.load_state_dict(args.high_model_path_sample80, map_location="cpu"))
    high_model_sample80.to(dist_util.dev()).eval()

    if args.use_fp16:
        low_model.convert_to_fp16()
        high_model.convert_to_fp16()
        high_model_sample80.convert_to_fp16()

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

    args.data = 'ETTh1'
    args.data_path = 'ETTh1.csv'
    
    _ , low_freq_data, high_freq_data = data_provider(args, flag=args.split)  # check
    _, data_origin = data_provider_origin(args, flag=args.split)  
    args.num_samples = len(low_freq_data.dataset)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_path = f"images_{timestamp}"
    for i, (low_data, high_data, data_original) in enumerate(zip(low_freq_data, high_freq_data, data_origin)):
        # 数据预处理（同原代码，只需重复一遍用于低频和高频）
        def process_input(data):
            conv_layer = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1).to(dist_util.dev()).double()
            x, y = data[0].unsqueeze(1).double().to(dist_util.dev()), data[1].unsqueeze(1).double().to(dist_util.dev())
            x, y = conv_layer(x), conv_layer(y)
            x, y = F.pad(x, (0, 25)), F.pad(y, (0, 25))
            return x, y

        x0_low, y0_low = process_input(low_data)
        x0_high, y0_high = process_input(high_data)
        x0, y0 = data_original[0].unsqueeze(1).double().to(dist_util.dev()), data_original[1].unsqueeze(1).double().to(dist_util.dev())
        # model_kwargs = {'xT': y0}

        # reduce_channels = nn.Conv2d(32, 6, kernel_size=1)
        # reduce_channels = reduce_channels.to(dist_util.dev()).double()
        # y0_reduced = reduce_channels(y0)
        # y0_reduced = y0_reduced.to(dist_util.dev())
        # print("y0_reduced shape: ",y0_reduced.shape)

        sample_low, _, nfe_low = karras_sample(
            low_diffusion, low_model, y0_low, x0_low,
            steps=args.steps, model_kwargs={'xT': y0_low},
            device=dist_util.dev(), clip_denoised=args.clip_denoised,
            sampler=args.sampler, sigma_min=low_diffusion.sigma_min,
            sigma_max=low_diffusion.sigma_max, churn_step_ratio=args.churn_step_ratio,
            rho=args.rho, guidance=args.guidance
        )
        sample_high, _, nfe_high= karras_sample(
            high_diffusion, high_model, y0_high, x0_high,
            steps=args.steps, model_kwargs={'xT': y0_high},
            device=dist_util.dev(), clip_denoised=args.clip_denoised,
            sampler=args.sampler, sigma_min=high_diffusion.sigma_min,
            sigma_max=high_diffusion.sigma_max, churn_step_ratio=args.churn_step_ratio,
            rho=args.rho, guidance=args.guidance
        )
        sample_high_sample80, _, nfe_high_sample80= karras_sample(
            high_diffusion_sample80, high_model_sample80, y0_high, x0_high,
            steps=args.steps, model_kwargs={'xT': y0_high},
            device=dist_util.dev(), clip_denoised=args.clip_denoised,
            sampler=args.sampler, sigma_min=high_diffusion.sigma_min,
            sigma_max=high_diffusion.sigma_max, churn_step_ratio=args.churn_step_ratio,
            rho=args.rho, guidance=args.guidance
        )
        sample_low = sample_low.detach().cpu().numpy()
        sample_high = sample_high.detach().cpu().numpy()
        #fusion
        sample = sample_low + sample_high

        # visualization
        # x0_image = x0_high.detach().cpu().numpy()
        # y0_image = y0_high.detach().cpu().numpy()
        # sample = th.tensor(sample)  # 转回 tensor
        # x0_image = th.tensor(x0_image)  # 转回 tensor
        # y0_image = th.tensor(y0_image)  # 转回 tensor
        # num_display = min(32, sample.shape[0])
        # if i == 0 and dist.get_rank() == 0:
        #     plot_time_series_batch(sample, f'{sample_dir}/sample_{i}.png', focus_f=0, ground_truth=y0_image)
        #     if x0_image is not None:
        #         plot_time_series_batch(x0_image, f'{sample_dir}/x_{i}.png', focus_f=0)
        #     plot_time_series_batch(y0_image, f'{sample_dir}/y_{i}.png', focus_f=0)

        # sys.exit(0)

        
        logger.log("sample.shape:", sample.shape)
        sample = th.tensor(sample)  # 转回 tensor
        sample = sample.mean(dim=1, keepdim=True)  # 取平均值恢复单通道
        sample_high = th.tensor(sample_high)  # 转回 tensor
        sample_high = sample_high.mean(dim=1, keepdim=True)  # 取平均值恢复单通道
        sample_low = th.tensor(sample_low)  # 转回 tensor   
        sample_low = sample_low.mean(dim=1, keepdim=True)  # 取平均值恢复单通道
        # sample_high_sample80 = th.tensor(sample_high_sample80)  # 转回 tensor
        sample_high_sample80 = sample_high_sample80.mean(dim=1, keepdim=True)  # 取平均值恢复单通道
        logger.log("After: sample shape: ", sample.shape)   
        # sample = sample.contiguous()
        

        batch_idx = 0  # 只画 batch 内的第一个样本
        feature_idx = 0  # 选取第一个 feature 进行可视化

        context_len = x0.shape[2]  # 过去的时间步
        horizon_len = y0.shape[2]  # 未来的时间步
        label_len = 48

        # 取出 batch 内的某个样本，并从 3 通道数据恢复成单通道
        x0_plot = x0[batch_idx, 0, :, feature_idx].detach().cpu().numpy()
        y0_plot = y0[batch_idx, 0, :, feature_idx].detach().cpu().numpy()
        sample_plot = sample[batch_idx, 0, :, feature_idx]
        sample_low_plot = sample_low[batch_idx, 0, :, feature_idx]
        sample_high_plot = sample_high[batch_idx, 0, :, feature_idx].detach().cpu().numpy()
        sample_high_sample80_plot = sample_high_sample80[batch_idx, 0, :, feature_idx].detach().cpu().numpy()

        y0 = y0.squeeze(1)  # 去掉通道维度
        fft_tool = Two_dimensional_FFT_FAMnet.TwoDimensionalFFT(device='cuda')
        low_y0_freq, high_y0_freq = fft_tool.filter_frequency_bands_two_freq(y0, cutoff=0.5)  # numpy 格式
        low_y0_plot = low_y0_freq[batch_idx,  :, feature_idx].detach().cpu().numpy()
        high_y0_plot = high_y0_freq[batch_idx,  :, feature_idx].detach().cpu().numpy()

        # 转为 tensor
        low_y0_freq = th.tensor(low_y0_freq)
        high_y0_freq = th.tensor(high_y0_freq)

        # 构造时间轴
        context_time = np.arange(context_len)  # 过去的时间步
        horizon_time = np.arange(context_len - label_len, context_len + horizon_len - label_len)  # 未来的时间步

        # 画图
        # print("start plot...")
        # plt.figure(figsize=(10, 5))
        # plt.plot(context_time, x0_plot, label="Context (Input)", color="blue")
        # plt.plot(horizon_time, y0_plot, label="Ground Truth (Future)", color="green")
        # plt.plot(horizon_time, sample_plot, label="Predicted (Future)", color="red", linestyle="dashed")
        # plt.plot(horizon_time, sample_low_plot, label="Predicted (Future) low", color="orange", linestyle="dashed")
        # plt.plot(horizon_time, sample_high_plot, label="Predicted (Future) high", color="purple", linestyle="dashed")
        # plt.plot(horizon_time, low_y0_plot, label="Ground Truth (Low Freq)", color="darkgreen", linestyle="dotted")
        # plt.plot(horizon_time, high_y0_plot, label="Ground Truth (High Freq)", color="brown", linestyle="dotted")

        # plt.axvline(x=context_len - label_len - 1, color="black", linestyle="dotted")  # 分割过去和未来
        # plt.xlabel("Time Steps")
        # plt.ylabel("Value")
        # plt.legend()

       
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # # 检查文件夹是否存在，若不存在则创建
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)
        # save_path = os.path.join(folder_path, f"lstf_ddbm_{timestamp}.png")
        # plt.savefig(save_path, dpi=300, bbox_inches="tight")
        # plt.show()
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        plt.figure(figsize=(12, 5))
        plt.plot(context_time, x0_plot, label="Context (Input)", color="blue")
        plt.plot(horizon_time, y0_plot, label="Ground Truth (Future)", color="red")
        plt.plot(horizon_time, high_y0_plot, label="Ground Truth (High Freq)", color="brown")
        plt.plot(horizon_time, sample_high_sample80_plot, label='High Model (sample300)', color='blue')
        plt.plot(horizon_time, sample_high_plot, label='High Model (sample80)', color='red', linestyle='--')
        plt.title("High-Frequency Model Prediction Comparison")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(folder_path, f"comparison_high_model_{i}.png"))
        plt.close()    
        sys.exit(0)

        print("start plot low frequency...")
        plt.figure(figsize=(10, 5))
        plt.plot(context_time, x0_plot, label="Context (Input)", color="blue")
        plt.plot(horizon_time, y0_plot, label="Ground Truth (Future)", color="red")
        plt.plot(horizon_time, low_y0_plot, label="Ground Truth (Low Freq)", color="darkgreen")
        plt.plot(horizon_time, sample_low_plot, label="Predicted (Low Freq)", color="darkgreen", linestyle="dashed")

        plt.axvline(x=context_len - label_len - 1, color="black", linestyle="dotted")  # 分割过去和未来
        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        plt.legend()   
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(folder_path, f"lstf_ddbm_low_freq_{timestamp}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show() 

        print("start plot high frequency...")
        plt.figure(figsize=(10, 5))
        plt.plot(context_time, x0_plot, label="Context (Input)", color="blue")
        plt.plot(horizon_time, y0_plot, label="Ground Truth (Future)", color="red")
        plt.plot(horizon_time, high_y0_plot, label="Ground Truth (High Freq)", color="brown")
        plt.plot(horizon_time, sample_high_plot, label="Predicted (High Freq)", color="brown", linestyle="dashed")
        plt.axvline(x=context_len - label_len - 1, color="black", linestyle="dotted")  # 分割过去和未来
        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        plt.legend()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(folder_path, f"lstf_ddbm_high_freq_{timestamp}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        # all_images.append(gathered_samples.detach().cpu().numpy())
        
        
    logger.log(f"created {len(all_images) * args.batch_size * dist.get_world_size()} samples")
        

    arr = np.concatenate(all_images, axis=0)
    arr = arr[:args.num_samples]
    
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(sample_dir, f"samples_{shape_str}_nfe{nfe_high}.npz")
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
        low_model_path="/home/zzx/projects/rrg-timsbc/zzx/LTSF_DDBM/diffusion/workdir/ett256_256d_2025-04-08_09-17-21/ema_0.9999_100000.pt",
        high_model_path='/home/zzx/projects/rrg-timsbc/zzx/LTSF_DDBM/diffusion/workdir/ett256_256d_high_freq_2025-04-10_21-44-23/ema_0.9999_100000.pt',
        high_model_path_sample80='/home/zzx/projects/rrg-timsbc/zzx/LTSF_DDBM/diffusion/workdir/ett256_256d_high_freq_2025-04-08_11-09-48/ema_0.9999_100000.pt',
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
