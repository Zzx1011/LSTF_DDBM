from scipy import fft
import torch
import numpy as np  
import matplotlib.pyplot as plt
class TwoDimensionalFFT(torch.nn.Module):
    """
    A class to perform 2D FFT operations on tensors.
    """

    def __init__(self, device='cuda'):
        """
        Initializes the TwoDimensionalFFT class.

        Args:
            device (str): The device to perform computations on ('cuda' or 'cpu').
        """
        self.device = device
        super(TwoDimensionalFFT, self).__init__()
    
    def forward(self, tensor, cutoff=0.2):
        """
        Forward method to perform 2D FFT and filter frequency bands.

        Args:
            tensor (torch.Tensor): The input tensor to be filtered.
            cutoff (float, optional): The cutoff value for frequency band filtering.

        Returns:
            torch.Tensor: The low frequency band of the input tensor.
            torch.Tensor: The mid frequency band of the input tensor.
            torch.Tensor: The high frequency band of the input tensor.
        """
        return self.filter_frequency_bands(tensor, cutoff)

    def reshape_to_square(self, tensor):
        """
        Reshapes a tensor to a square shape.

        Args:
            tensor (torch.Tensor): The input tensor of shape (B, C, N), where B is the batch size,
                C is the number of channels, and N is the number of elements.

        Returns:
            tuple: A tuple containing:
                - square_tensor (torch.Tensor): The reshaped tensor of shape (B, C, side_length, side_length),
                  where side_length is the length of each side of the square tensor.
                - side_length (int): The length of each side of the square tensor.
                - side_length (int): The length of each side of the square tensor.
                - N (int): The original number of elements in the input tensor.
        """
        B, C, N = tensor.shape
        side_length = int(np.ceil(np.sqrt(N)))
        padded_length = side_length ** 2
        
        padded_tensor = torch.zeros((B, C, padded_length), device=tensor.device)
        padded_tensor[:, :, :N] = tensor

        square_tensor = padded_tensor.view(B, C, side_length, side_length)
        
        return square_tensor, side_length, side_length, N
    
    def filter_frequency_bands(self, tensor, cutoff=0.2):
                """
                Filters the input tensor into low, mid, and high frequency bands.

                Args:
                    tensor (torch.Tensor): The input tensor to be filtered.
                    cutoff (float, optional): The cutoff value for frequency band filtering.

                Returns:
                    torch.Tensor: The low frequency band of the input tensor.
                    torch.Tensor: The mid frequency band of the input tensor.
                    torch.Tensor: The high frequency band of the input tensor.
                """

                tensor = tensor.float()
                tensor, H, W, N = self.reshape_to_square(tensor)
                B, C, _, _ = tensor.shape

                max_radius = np.sqrt((H // 2)**2 + (W // 2)**2)
                low_cutoff = max_radius * cutoff
                high_cutoff = max_radius * (1 - cutoff)

                fft_tensor = torch.fft.fftshift(torch.fft.fft2(tensor, dim=(-2, -1)), dim=(-2, -1))

                def create_filter(shape, low_cutoff, high_cutoff, mode='band', device=self.device):
                    rows, cols = shape
                    center_row, center_col = rows // 2, cols // 2
                    
                    y, x = torch.meshgrid(torch.arange(rows, device=device), torch.arange(cols, device=device), indexing='ij')
                    distance = torch.sqrt((y - center_row) ** 2 + (x - center_col) ** 2)
                    
                    mask = torch.zeros((rows, cols), dtype=torch.float32, device=device)

                    if mode == 'low':
                        mask[distance <= low_cutoff] = 1
                    elif mode == 'high':
                        mask[distance >= high_cutoff] = 1
                    elif mode == 'band':
                        mask[(distance > low_cutoff) & (distance < high_cutoff)] = 1
                    
                    return mask

                low_pass_filter = create_filter((H, W), low_cutoff, None, mode='low')[None, None, :, :]
                high_pass_filter = create_filter((H, W), None, high_cutoff, mode='high')[None, None, :, :]
                mid_pass_filter = create_filter((H, W), low_cutoff, high_cutoff, mode='band')[None, None, :, :]

                low_freq_fft = fft_tensor * low_pass_filter
                high_freq_fft = fft_tensor * high_pass_filter
                mid_freq_fft = fft_tensor * mid_pass_filter

                low_freq_tensor = torch.fft.ifft2(torch.fft.ifftshift(low_freq_fft, dim=(-2, -1)), dim=(-2, -1)).real
                high_freq_tensor = torch.fft.ifft2(torch.fft.ifftshift(high_freq_fft, dim=(-2, -1)), dim=(-2, -1)).real
                mid_freq_tensor = torch.fft.ifft2(torch.fft.ifftshift(mid_freq_fft, dim=(-2, -1)), dim=(-2, -1)).real

                low_freq_tensor = low_freq_tensor.view(B, C, H * W)[:, :, :N]
                high_freq_tensor = high_freq_tensor.view(B, C, H * W)[:, :, :N]
                mid_freq_tensor = mid_freq_tensor.view(B, C, H * W)[:, :, :N]

                return low_freq_tensor, mid_freq_tensor, high_freq_tensor
    
    def filter_frequency_bands_two_freq(self, tensor, cutoff=0.5):
        """
        Filters the input tensor into low and high frequency bands.

        Args:
            tensor (torch.Tensor): The input tensor to be filtered.
            cutoff (float): The cutoff value for separating low and high frequency.

        Returns:
            torch.Tensor: The low frequency band of the input tensor.
            torch.Tensor: The high frequency band of the input tensor.
        """
        tensor = tensor.float()
        tensor, H, W, N = self.reshape_to_square(tensor)
        B, C, _, _ = tensor.shape

        max_radius = np.sqrt((H // 2)**2 + (W // 2)**2)
        low_cutoff = max_radius * cutoff
        high_cutoff = max_radius * (1 - cutoff)

        fft_tensor = torch.fft.fftshift(torch.fft.fft2(tensor, dim=(-2, -1)), dim=(-2, -1))

        def create_filter(shape, cutoff, mode='low', device=self.device):
            rows, cols = shape
            center_row, center_col = rows // 2, cols // 2
            y, x = torch.meshgrid(torch.arange(rows, device=device), torch.arange(cols, device=device), indexing='ij')
            distance = torch.sqrt((y - center_row) ** 2 + (x - center_col) ** 2)
            mask = torch.zeros((rows, cols), dtype=torch.float32, device=device)

            if mode == 'low':
                mask[distance <= cutoff] = 1
            elif mode == 'high':
                mask[distance >= cutoff] = 1
            return mask

        low_pass_filter = create_filter((H, W), low_cutoff, mode='low')[None, None, :, :]
        high_pass_filter = create_filter((H, W), high_cutoff, mode='high')[None, None, :, :]

        low_freq_fft = fft_tensor * low_pass_filter
        high_freq_fft = fft_tensor * high_pass_filter

        def radial_average(fft_data):
            """Compute 1D radial average of 2D FFT magnitude."""
            H, W = fft_data.shape
            center = (H // 2, W // 2)

            y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            r = torch.sqrt((x - center[1])**2 + (y - center[0])**2)
            r = r.to(torch.int32).flatten()

            mag = torch.abs(fft_data).flatten()
            max_r = r.max().item() + 1

            radial_mean = torch.zeros(max_r, dtype=torch.float32, device=fft_data.device)
            counts = torch.zeros(max_r, dtype=torch.float32, device=fft_data.device)

            for i in range(len(r)):
                radial_mean[r[i]] += mag[i]
                counts[r[i]] += 1

            radial_mean /= (counts + 1e-8)
            return radial_mean[:min(H, W)//2]  # 保证对称性，只取一半频率

        # 只绘制第一个样本第一个通道
        b, c = 0, 0
        r_orig = radial_average(fft_tensor[b, c]).cpu().numpy()
        r_low = radial_average(low_freq_fft[b, c]).cpu().numpy()
        r_high = radial_average(high_freq_fft[b, c]).cpu().numpy()
        freqs = np.linspace(0, 1, len(r_orig))  # 归一化频率

        plt.figure(figsize=(8, 5))
        plt.plot(freqs, r_orig, label='Original FFT')
        # plt.plot(freqs, r_low, label='Low Frequency')
        # plt.plot(freqs, r_high, label='High Frequency')
        plt.plot(freqs, r_low+r_high, label='added', color='blue')
        plt.title("1D Frequency Magnitude Spectrum (Radial Avg)")
        plt.xlabel("Normalized Frequency")
        plt.ylabel("Magnitude")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("/home/zzx/projects/rrg-timsbc/zzx/LTSF_DDBM/data_provider/low_high_freq.png", dpi=300, bbox_inches="tight")
        plt.show()

        low_freq_tensor = torch.fft.ifft2(torch.fft.ifftshift(low_freq_fft, dim=(-2, -1)), dim=(-2, -1)).real
        high_freq_tensor = torch.fft.ifft2(torch.fft.ifftshift(high_freq_fft, dim=(-2, -1)), dim=(-2, -1)).real
        
        low_freq_tensor = low_freq_tensor.view(B, C, H * W)[:, :, :N]
        high_freq_tensor = high_freq_tensor.view(B, C, H * W)[:, :, :N]

        return low_freq_tensor, high_freq_tensor
    
def main():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        fft_processor = TwoDimensionalFFT(device=device)
        
        # Create a random tensor for demonstration
        tensor = torch.randn(2, 3, 64 * 64).to(device)  # Shape: (B, C, N)
        cutoff = 0.5
        low_freq, high_freq = fft_processor.filter_frequency_bands_two_freq(tensor, cutoff=cutoff)
        print("Low Frequency Shape:", low_freq.shape)
        print("High Frequency Shape:", high_freq.shape)

if __name__ == "__main__":  
        main()