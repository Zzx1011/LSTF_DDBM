import sys
sys.path.append('/home/zzx/projects/rrg-timsbc/zzx/LTSF_DDBM')
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader
from data_provider.Two_dimensional_FFT_FAMnet import TwoDimensionalFFT
import torch
from torch.utils.data import TensorDataset

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}

import numpy as np


def adaptive_frequency_selection(x, gamma):
    # 对时间序列数据进行傅里叶变换
    x_freq = np.fft.fft(x)
    # 计算频谱密度（这里简单用幅值平方表示，实际可能更复杂）
    spectral_density = np.abs(x_freq) ** 2
    total_spectral_info = np.sum(spectral_density)
    # 计算每个频率信息占总频谱信息的比例
    info_ratio = spectral_density / total_spectral_info
    cumulative_ratio = np.cumsum(info_ratio)
    kappa = np.argwhere(cumulative_ratio >= gamma)[0][0]
    # 划分低频和高频成分
    x_low_freq = np.pad(x_freq[:kappa + 1], (0, len(x_freq) - kappa - 1), 'constant')
    x_high_freq = np.pad(x_freq[kappa + 1:], (kappa + 1, 0), 'constant')
    return x_low_freq, x_high_freq


def data_provider(args, flag):
    # 设置默认参数
    # args.data = 'ETTh1'
    args.root_path = '/home/zzx/projects/rrg-timsbc/zzx/LTSF_DDBM/data/ETT'
    # args.data_path = 'ETTh1.csv'
    args.seq_len = 96
    args.label_len = 48
    args.pred_len = 48 # modified, 96 original
    args.features = 'M'
    args.target = 'OT'
    args.embed = 'timeF'
    args.freq = 'h'
    args.batch_size = 128 #modified, 32 original
    args.num_workers = 10
    args.train_only = False


    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    train_only = args.train_only

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        train_only=train_only
    )
    print(flag, len(data_set))
    
    # data_loader = DataLoader(
    #     data_set,
    #     batch_size=batch_size,
    #     shuffle=shuffle_flag,
    #     num_workers=args.num_workers,
    #     drop_last=drop_last)
    # return data_set, data_loader

    # # 新增高低频分离部分
    # gamma = args.gamma  # 可根据实际情况调整
    # all_low_freq_data = []
    # all_high_freq_data = []
    # for i in range(len(data_set)):
    #     data = data_set[i][0]  # 假设数据在第一个位置，根据实际情况调整
    #     if len(data.shape) > 1:  # 如果是多维数据，假设每一维都要进行高低频分离
    #         low_freq_list = []
    #         high_freq_list = []
    #         for dim in range(data.shape[1]):
    #             low_freq, high_freq = adaptive_frequency_selection(data[:, dim], gamma)
    #             low_freq_list.append(low_freq)
    #             high_freq_list.append(high_freq)
    #         low_freq_data = np.array(low_freq_list).transpose()
    #         high_freq_data = np.array(high_freq_list).transpose()
    #     else:
    #         low_freq_data, high_freq_data = adaptive_frequency_selection(data, gamma)
    #     all_low_freq_data.append(low_freq_data)
    #     all_high_freq_data.append(high_freq_data)
    # all_low_freq_data = np.array(all_low_freq_data)
    # all_high_freq_data = np.array(all_high_freq_data)

    # low_freq_data_loader = DataLoader(
    #     all_low_freq_data,
    #     batch_size=batch_size,
    #     shuffle=shuffle_flag,
    #     num_workers=args.num_workers,
    #     drop_last=drop_last)
    # high_freq_data_loader = DataLoader(
    #     all_high_freq_data,
    #     batch_size=batch_size,
    #     shuffle=shuffle_flag,
    #     num_workers=args.num_workers,
    #     drop_last=drop_last)

    # return data_set, low_freq_data_loader, high_freq_data_loader

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 原始数据张量化处理
    all_x = []
    all_y = []
    for i in range(len(data_set)):
        sample = data_set[i]
        x = torch.tensor(sample[0])  # Convert from numpy to torch
        y = torch.tensor(sample[1])
        all_x.append(x.unsqueeze(0))
        all_y.append(y.unsqueeze(0))
    x_tensor = torch.cat(all_x, dim=0).to(device)  # [B, C, T]
    y_tensor = torch.cat(all_y, dim=0).to(device)

#     # 将 1D 序列重塑成 2D 以使用 FFT（例如：[B, C, T] → [B, C, H, W]）
#     B, C, T = x_tensor.shape
#     H = int(np.sqrt(T))
#     W = H if H * H == T else H + 1
#     x_tensor_padded = torch.zeros((B, C, H, W), device='cuda')
#    # 1. Flatten x_tensor to 2D
#     B, C, T = x_tensor.shape
#     HW = int(np.ceil(np.sqrt(T)))
#     total_size = HW * HW

#     # 2. Padding if needed
#     if T < total_size:
#         pad_len = total_size - T
#         x_tensor = torch.nn.functional.pad(x_tensor, (0, pad_len), mode='constant', value=0)

#     # 3. Reshape to square [B, C, H, W]
#     x_tensor_padded = x_tensor.view(B, C, HW, HW)

    
    # 调用 FFT 模块
    fft_processor = TwoDimensionalFFT(device=device)
    # low, mid, high = fft_processor.filter_frequency_bands(x_tensor)
    low, high = fft_processor.filter_frequency_bands_two_freq(x_tensor, cutoff=0.5)

    # sys.exit(0)

    # 构造 DataLoader：每个频段都用同一个目标 y_tensor
    def build_loader(freq_tensor):
        dataset = TensorDataset(freq_tensor.detach().cpu(), y_tensor.detach().cpu())
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_flag, num_workers=args.num_workers, drop_last=drop_last)

    low_loader = build_loader(low)
    # mid_loader = build_loader(mid)
    high_loader = build_loader(high)

    return data_set, low_loader, high_loader

def data_provider_origin(args, flag):
    # 设置默认参数
    # args.data = 'ETTh1'
    args.root_path = '/home/zzx/projects/rrg-timsbc/zzx/LTSF_DDBM/data/ETT'
    # args.data_path = 'ETTh1.csv'
    args.seq_len = 96
    args.label_len = 48
    args.pred_len = 48 # modified, 96 original
    args.features = 'M'
    args.target = 'OT'
    args.embed = 'timeF'
    args.freq = 'h'
    args.batch_size = 32
    args.num_workers = 10
    args.train_only = False


    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    train_only = args.train_only

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        train_only=train_only
    )
    print(flag, len(data_set))
    
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader