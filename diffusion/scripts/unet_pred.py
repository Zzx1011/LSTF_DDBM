import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys
sys.path.append('/home/zzx/projects/rrg-timsbc/zzx')

from LTSF_DDBM.data_provider.data_factory import data_provider, data_provider_origin
from LTSF_DDBM.diffusion.ddbm.unet_conv1d import UNetConv1DModel  
from LTSF_DDBM.diffusion.ddbm.unet import UNetModel
from  LTSF_DDBM.diffusion.ddbm import logger

def train_unet(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dataset_name, filename in zip(['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'],
                                      ['ETTh1.csv', 'ETTh2.csv', 'ETTm1.csv', 'ETTm2.csv']):
        logger.log(f"Training on dataset: {dataset_name}")
        args.data = dataset_name
        args.data_path = filename
        args.freq = 'h' if 'ETTh' in dataset_name else 'm'

        # _, low_freq_data, high_freq_data = data_provider(args, flag='train')
        # train_loader = low_freq_data
        _, data = data_provider_origin(args, flag='train')
        train_loader = data
        

        # Infer input/output dimensions
        x_sample, y_sample, _, _= next(iter(train_loader))
        input_dim = x_sample.shape[1]
        output_dim = y_sample.shape[1]
        window_size = x_sample.shape[2]

        model = UNetConv1DModel(
            image_size=window_size,
            in_channels=input_dim,
            model_channels=256,
            out_channels=output_dim,
            num_res_blocks=3,  # 2
            attention_resolutions=[4, 8],
            dropout=0.1,
            channel_mult=(1, 2, 4),
            # channel_mult=(1, 1, 2, 2, 4, 4),
            conv_resample=True,
            dims=1,
            num_classes=None,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=4,
            num_head_channels=64,
            resblock_updown=True,
            use_new_attention_order=False,
            attention_type='vanilla',
            condition_mode=None,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(args.epochs):
            epoch_loss = 0
            for x, y,_, _ in train_loader:
                x, y = F.pad(x, (0, 25)), F.pad(y, (0, 25))
                # x = x.permute(0, 2, 1)  # (B, T, F) → (B, F, T)
                # y = y.permute(0, 2, 1)  # (B, T, F) → (B, F, T)
                print("x.shape:",x.shape)   
                print("y.shape:",y.shape)
                # print("x:",x)
                # print("y:",y)
                x = x.float().to(device)
                y = y.float().to(device)
                timesteps = torch.zeros(x.size(0), dtype=torch.long).float().to(device)

                pred = model(x, timesteps)
                loss = criterion(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(f"[{dataset_name} | Epoch {epoch+1}] Loss: {epoch_loss/len(train_loader):.4f}")

        model_path = f"/home/zzx/projects/rrg-timsbc/zzx/LTSF_DDBM/diffusion/unet1d_{dataset_name}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Saved model to {model_path}")

        evaluate_unet(args, model_path, dataset_name)


def evaluate_unet(args, model_path, dataset_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.data = dataset_name
    args.data_path = f"{dataset_name}.csv"
    args.freq = 'h' if 'ETTh' in dataset_name else 'm'
    # _, low_freq_data, high_freq_data = data_provider(args, flag='test')
    # test_loader = low_freq_data
    _, data = data_provider_origin(args, flag='test')
    test_loader = data
    print("test_loader:",len(test_loader))

    x_sample, y_sample, _, _= next(iter(test_loader))
    input_dim = x_sample.shape[1]
    output_dim = y_sample.shape[1]
    window_size = x_sample.shape[2]

    model = UNetConv1DModel(
        image_size=window_size,
        in_channels=input_dim,
        model_channels=256,
        out_channels=output_dim,
        num_res_blocks=3,  # 2
        attention_resolutions=[4, 8],
        dropout=0.1,
        channel_mult=(1, 2, 4),
        # channel_mult=(1, 1, 2, 2, 4, 4),
        conv_resample=True,
        dims=1,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=4,
        num_head_channels=64,
        resblock_updown=True,
        use_new_attention_order=False,
        attention_type='vanilla',
        condition_mode=None,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"[INFO] Loaded model from {model_path}")

    all_preds = []
    all_trues = []
    with torch.no_grad():
        for x, y ,_, _ in test_loader:
            x, y = F.pad(x, (0, 25)), F.pad(y, (0, 25))
            # x = x.permute(0, 2, 1)  # (B, T, F) → (B, F, T)
            # y = y.permute(0, 2, 1)  # (B, T, F) → (B, F, T)
            x = x.to(device)
            y = y.to(device)
            timesteps = torch.zeros(x.size(0), dtype=torch.long).to(device)
            pred = model(x, timesteps)
            # break

            all_preds.append(pred.cpu())
            all_trues.append(y.cpu())

        # 拼接所有结果
        all_preds = torch.cat(all_preds, dim=0).numpy()  # shape: [N, C, T]
        all_trues = torch.cat(all_trues, dim=0).numpy()
        print(f"all_preds.shape: {all_preds.shape}")
        print(f"all_trues.shape: {all_trues.shape}")

        num_features = 7 # 7
        maes = []
        mses = []
        for i in range(num_features):
            mae = mean_absolute_error(all_trues[:, :, i].flatten(), all_preds[:, :, i].flatten())
            mse = mean_squared_error(all_trues[:, :, i].flatten(), all_preds[:, :, i].flatten())
            maes.append(mae)
            mses.append(mse)
            print(f"[Feature {i}] MAE: {mae:.6f}, MSE: {mse:.6f}")

        # reshape 为 (N * C * T) 进行整体 MAE/MSE 计算
        mse = mean_squared_error(all_trues.flatten(), all_preds.flatten())
        mae = mean_absolute_error(all_trues.flatten(), all_preds.flatten())
        print(f"[EVAL] {dataset_name} - MSE: {mse:.6f}, MAE: {mae:.6f}")
       

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_idx = 0
    feature_idx = 0

    x_np = x[batch_idx, :, feature_idx].cpu().numpy()
    y_np = y[batch_idx, :, feature_idx].cpu().numpy()
    pred_np = pred[batch_idx, :, feature_idx].cpu().numpy()
    print(f"x_np.shape: {x_np.shape}")
    print(f"y_np.shape: {y_np.shape}")
    print(f"pred_np.shape: {pred_np.shape}")

    context_len = x.shape[1]
    horizon_len = y.shape[1]

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(context_len), x_np, label="Context (Input)", color="blue")
    plt.plot(np.arange(context_len, context_len + horizon_len), y_np, label="Ground Truth (Future)", color="green")
    plt.plot(np.arange(context_len, context_len + horizon_len), pred_np, label="Predicted (Future)", color="red", linestyle="dashed")
    plt.axvline(x=context_len - 1, color="black", linestyle="dotted")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.legend()

    save_path = f"unet_pred_{dataset_name}_{timestamp}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved prediction plot to {save_path}")

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--dataset", type=str, default="ETTh1")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser

def main():
    args = create_argparser().parse_args()
    # model = train_unet(args)
    for dataset_name in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']:
        model_path = f"/home/zzx/projects/rrg-timsbc/zzx/LTSF_DDBM/diffusion/unet1d_{dataset_name}.pth"
        evaluate_unet(args, model_path, dataset_name)
    

if __name__ == "__main__":
    main()
