import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import welch


def compute_fft(data, sampling_rate):
    # Based on https://github.com/kovalalvi/beira
    """
    Return (freqs, amplitudes)
    """
    n = data.shape[0]
    values = torch.fft.rfft(data, n=n, norm ='forward')[:n // 2]  # forward means 1/n
    freqs = torch.fft.fftfreq(n=n, d=1 / sampling_rate)[:values.shape[0]]
    return freqs, values


class SpatialFilter(nn.Module):
    def __init__(self, n_eeg_channels, n_branches, spatial_bias=True, apply_batchnorm=True):
        super().__init__()
        self.n_eeg_channels = n_eeg_channels
        self.n_branches = n_branches
        self.spatial_bias = spatial_bias
        self.apply_batchnorm = apply_batchnorm

        self.spatial_filter = nn.Conv1d(
            in_channels=n_eeg_channels,
            out_channels=n_branches,
            kernel_size=1,
            bias=spatial_bias
        )
        if apply_batchnorm:
            self.batchnorm_spatial = nn.BatchNorm1d(n_branches, affine=False)

    def forward(self, x):
        x = self.spatial_filter(x)
        if self.apply_batchnorm:
            x = self.batchnorm_spatial(x)
        return x
    

class TemporalFilter(nn.Module):
    def __init__(self, n_branches, filter_length, bias, nonlinearity_kind, apply_batchnorm=True, padding='same'):
        super().__init__()
        self.n_branches = n_branches
        self.filter_length = filter_length
        self.bias = bias
        self.nonlinearity_kind = nonlinearity_kind
        self.apply_batchnorm = apply_batchnorm
        self.padding = padding

        self.temporal_filter = nn.Conv1d(
            in_channels=n_branches,
            out_channels=n_branches,
            kernel_size=filter_length,
            groups=n_branches,
            padding=padding,
            bias=bias
        )

        if apply_batchnorm:
            self.batchnorm = nn.BatchNorm1d(n_branches, affine=False)

        if nonlinearity_kind == 'relu':
            self.nonlinearity = nn.ReLU()
        elif nonlinearity_kind == 'abs':
            self.nonlinearity = nn.LeakyReLU(negative_slope=-1)
        elif nonlinearity_kind is None:
            self.nonlinearity = None
        else:
            raise ValueError(f'Unknown nonlinearity kind: {nonlinearity_kind}')

    def get_output_length(self, input_length):
        if self.padding == 'valid':
            padding = 0
        else:
            padding = self.padding
        if self.padding == 'same':
            return input_length
        else:
            return (input_length + 2 * padding - self.filter_length + 1)
    
    def forward(self, x):
        x = self.temporal_filter(x)
        if self.apply_batchnorm:
            x = self.batchnorm(x)
        if self.nonlinearity_kind is not None:
            x = self.nonlinearity(x)
        return x
    

class SpatioTemporalBlock(nn.Module):
    def __init__(
            self, 
            n_eeg_channels, n_branches,
            temporal_length, 
            spatial_bias=True, temporal_bias=False,
            spatial_batch_norm=True, temporal_batch_norm=True,
            temporal_nonlinearity='abs',
            padding='same'
        ):
        super().__init__()
        self.n_eeg_channels = n_eeg_channels
        self.n_branches = n_branches
        self.temporal_length = temporal_length
        self.spatial_bias = spatial_bias
        self.temporal_bias = temporal_bias
        self.spatial_batch_norm = spatial_batch_norm
        self.temporal_batch_norm = temporal_batch_norm
        self.temporal_nonlinearity = temporal_nonlinearity
        self.padding = padding

        # Initialize spatial and temporal filters
        self.spatial_filter = SpatialFilter(
            n_eeg_channels=n_eeg_channels,
            n_branches=n_branches,
            spatial_bias=spatial_bias,
            apply_batchnorm=spatial_batch_norm
        )

        self.temporal_filter = TemporalFilter(
            n_branches=n_branches, 
            filter_length=temporal_length, 
            bias=temporal_bias, 
            nonlinearity_kind=temporal_nonlinearity,
            apply_batchnorm=temporal_batch_norm, 
            padding=padding
        )

    def forward(self, x):
        x = self.spatial_filter(x)
        x = self.temporal_filter(x)
        return x
    
    def get_spatial_patterns(self, spatial_filters, temporal_filters, eeg_data):
        # Based on https://github.com/kovalalvi/beira
        eeg_data = eeg_data.unsqueeze(0)

        spatial_patterns = []
    
        for branch_idx in range(self.n_branches):
            spatial_filter = spatial_filters[branch_idx, :]
            # spatial_filter: (n_eeg_channels, )
            temporal_filter = temporal_filters[branch_idx, :, :]
            # temporal_filter: (1, temporal_length)

            # Temporally filter data 
            temporal_filter = temporal_filter.repeat(self.n_eeg_channels, 1, 1)
            # temporal_filter: (n_eeg_channels, 1, temporal_length)
            temporally_filtred_eeg = F.conv1d(eeg_data, temporal_filter, padding='same', groups=self.n_eeg_channels).squeeze(dim=0)
            # temporally_filtred_eeg: (n_eeg_channels, n_samples)

            # Calculate covariance matrix
            eeg_cov = torch.cov(temporally_filtred_eeg)
            # eeg_cov: (n_eeg_channels, n_eeg_channels)

            # Compute spatial patterns
            pattern = torch.matmul(eeg_cov, spatial_filter)
            # pattern: (n_eeg_channels, )

            spatial_patterns.append(pattern)

        spatial_patterns = torch.stack(spatial_patterns, dim=0)

        return spatial_patterns.detach().cpu().numpy()
    
    def get_temporal_patterns(self, spatial_filters, temporal_filters, eeg_data, eeg_sampling_rate):
        # Based on https://github.com/kovalalvi/beira
        temporal_patterns = []
        
        for branch_idx in range(self.n_branches):
            spatial_filter = spatial_filters[branch_idx, :]
            # spatial_filter: (n_eeg_channels, )
            temporal_filter = temporal_filters[branch_idx, :, :].squeeze(dim=0)
            # temporal_filter: (temporal_length, )
            
            # Spatially filter data  
            spatially_filtered_eeg = spatial_filter @ eeg_data
            # spatially_filtered_eeg: (n_samples, )
            
            fft_kernel = np.fft.rfft(temporal_filter.detach().cpu().numpy(), n=eeg_sampling_rate * 2)
            f_welch, Pxx_welch = welch(
                spatially_filtered_eeg.detach().cpu().numpy(), fs=eeg_sampling_rate,
                window='hann',
                nperseg=eeg_sampling_rate * 2,
                noverlap=eeg_sampling_rate,
                nfft=eeg_sampling_rate * 2,
                scaling='density'
            )
            assert Pxx_welch.shape[0] == 1, Pxx_welch.shape
            Pxx_welch = Pxx_welch[0, :]
            pattern = np.abs(fft_kernel * Pxx_welch)

            temporal_patterns.append(pattern)
            
        fft_freqs = np.fft.rfftfreq(n=eeg_sampling_rate * 2, d=1/eeg_sampling_rate)
        return fft_freqs, np.stack(temporal_patterns, axis=0)
    
    def calculate_patterns(self, eeg_data, eeg_sampling_rate):
        # eeg_data: (n_eeg_channels, n_samples)
        spatial_patterns = []
        temporal_patterns = []

        spatial_filters = self.spatial_filter.spatial_filter.weight.detach().cpu().squeeze(dim=-1)
        # spatial_filters: (n_branches, n_eeg_channels)
        temporal_filters = self.temporal_filter.temporal_filter.weight.detach().cpu()
        # temporal_filters: (n_branches, 1, temporal_length)

        spatial_patterns = self.get_spatial_patterns(
            spatial_filters=spatial_filters, 
            temporal_filters=temporal_filters, 
            eeg_data=eeg_data
        )

        fft_freqs, temporal_patterns = self.get_temporal_patterns(
            spatial_filters=spatial_filters, 
            temporal_filters=temporal_filters, 
            eeg_data=eeg_data,
            eeg_sampling_rate=eeg_sampling_rate
        )

        return spatial_patterns, temporal_patterns, fft_freqs


class DecimatePooling(nn.Module):
    def __init__(self, decimation_factor):
        super().__init__()
        self.decimation_factor = decimation_factor

    def forward(self, x):
        # x: (batch_size, n_branches, time_length)
        return x[:, :, ::self.decimation_factor]
    

class EnvelopeDetector(nn.Module):
    def __init__(
            self, 
            n_eeg_channels, n_branches, window_size,
            band_pass_length, low_pass_length, 
            spatial_bias=True, band_pass_bias=False, low_pass_bias=True, 
            spatial_batch_norm=True, band_pass_batch_norm=True, low_pass_batch_norm=True,
            band_pass_nonlinearity='abs', low_pass_nonlinearity='relu',
            padding='same',
            pooling_kernel=None, pooling_type=None
        ):
        super().__init__()
        self.n_eeg_channels = n_eeg_channels
        self.n_branches = n_branches
        self.band_pass_length = band_pass_length
        self.low_pass_length = low_pass_length
        self.spatial_bias = spatial_bias
        self.band_pass_bias = band_pass_bias
        self.low_pass_bias = low_pass_bias
        self.spatial_batch_norm = spatial_batch_norm
        self.padding = padding
        self.pooling_kernel = pooling_kernel
        self.pooling_type = pooling_type

        self.spatio_temporal_block = SpatioTemporalBlock(
            n_eeg_channels=n_eeg_channels, 
            n_branches=n_branches,
            temporal_length=band_pass_length, 
            spatial_bias=spatial_bias, 
            temporal_bias=band_pass_bias,
            spatial_batch_norm=spatial_batch_norm, 
            temporal_batch_norm=band_pass_batch_norm,
            temporal_nonlinearity=band_pass_nonlinearity,
            padding=padding
        )

        self.low_pass_filter = TemporalFilter(
            n_branches=n_branches, 
            filter_length=low_pass_length, 
            bias=low_pass_bias, 
            nonlinearity_kind=low_pass_nonlinearity, 
            apply_batchnorm=low_pass_batch_norm, 
            padding=padding
        )

        output_length = self.spatio_temporal_block.temporal_filter.get_output_length(window_size)
        output_length = self.low_pass_filter.get_output_length(output_length)
        
        if pooling_type is not None:
            if pooling_type == 'decimate':
                self.pooling = DecimatePooling(pooling_kernel)
            elif pooling_type == 'avg':
                self.pooling = nn.AvgPool1d(kernel_size=pooling_kernel, stride=pooling_kernel)
            elif pooling_type == 'max':
                self.pooling = nn.MaxPool1d(kernel_size=pooling_kernel, stride=pooling_kernel)
            output_length = output_length // pooling_kernel
        
        self.output_length = output_length

    def forward(self, x):
        # x: (batch_size, n_eeg_channels, window_size)
        spatio_temporal_out = self.spatio_temporal_block(x)
        low_pass_out = self.low_pass_filter(spatio_temporal_out)
        if self.pooling_type is not None:
            low_pass_out = self.pooling(low_pass_out)
        # low_pass_out: (batch_size, n_branches, output_length)
        return low_pass_out


class EEGSimpleNet(nn.Module):
    def __init__(
            self, 
            n_eeg_channels, n_output, n_branches, window_size,
            band_pass_length, low_pass_length, 
            spatial_bias=True, band_pass_bias=False, low_pass_bias=True, 
            spatial_batch_norm=True, band_pass_batch_norm=True, low_pass_batch_norm=True,
            band_pass_nonlinearity='abs', low_pass_nonlinearity='relu',
            padding='same',
            pooling_kernel=None, pooling_type=None,
            dropout_p=None
        ):
        super().__init__()
        self.n_eeg_channels = n_eeg_channels
        self.n_output = n_output
        self.n_branches = n_branches
        self.band_pass_length = band_pass_length
        self.low_pass_length = low_pass_length
        self.spatial_bias = spatial_bias
        self.band_pass_bias = band_pass_bias
        self.low_pass_bias = low_pass_bias
        self.spatial_batch_norm = spatial_batch_norm
        self.padding = padding
        self.pooling_kernel = pooling_kernel
        self.pooling_type = pooling_type
        self.dropout_p = dropout_p

        self.envelope_detector = EnvelopeDetector(
            n_eeg_channels=n_eeg_channels, n_branches=n_branches, window_size=window_size, 
            band_pass_length=band_pass_length, low_pass_length=low_pass_length, 
            spatial_bias=spatial_bias, band_pass_bias=band_pass_bias, 
            low_pass_bias=low_pass_bias, spatial_batch_norm=spatial_batch_norm, 
            band_pass_batch_norm=band_pass_batch_norm, 
            low_pass_batch_norm=low_pass_batch_norm, 
            band_pass_nonlinearity=band_pass_nonlinearity, 
            low_pass_nonlinearity=low_pass_nonlinearity, 
            padding=padding, pooling_kernel=pooling_kernel, pooling_type=pooling_type
        )

        if dropout_p is not None:
            self.dropout = nn.Dropout(p=dropout_p)
        
        self.flatten = nn.Flatten(start_dim=1)
        self.output_layer = nn.Linear(self.envelope_detector.output_length * n_branches, n_output)

    def forward(self, x):
        # x: (batch_size, n_eeg_channels, window_size)
        low_pass_out = self.envelope_detector(x)
        # low_pass_out: (batch_size, n_branches, output_length)
        low_pass_out = self.flatten(low_pass_out)
        if self.dropout_p is not None:
            low_pass_out = self.dropout(low_pass_out)
        output = self.output_layer(low_pass_out)
        # output: (batch_size, n_output)
        return output
