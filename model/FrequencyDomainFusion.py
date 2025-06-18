import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops.carafe import normal_init, xavier_init, carafe
from torch.utils.checkpoint import checkpoint
import warnings
import numpy as np

def initialize_weights(module, mean_val=0, std_val=1, bias_val=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean_val, std_val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias_val)

def create_2d_window(rows, cols):

    window_row = np.hamming(rows)
    window_col = np.hamming(cols)
    window_2d = np.outer(window_row, window_col)
    return window_2d

class FrequencyDomainFusion(nn.Module):
    def __init__(self,
                high_res_channels,
                low_res_channels,
                scale_ratio=1,
                low_freq_kernel_size=5,
                high_freq_kernel_size=3,
                upsampling_groups=1,
                feature_encoder_kernel=3,
                feature_encoder_dilation=1,
                bottleneck_channels=64,        
                corner_alignment=False,
                interpolation_mode='nearest',
                use_high_pass=True,
                **kwargs):
        super().__init__()
        self.scale_ratio = scale_ratio
        self.low_freq_kernel_size = low_freq_kernel_size
        self.high_freq_kernel_size = high_freq_kernel_size
        self.upsampling_groups = upsampling_groups
        self.feature_encoder_kernel = feature_encoder_kernel
        self.feature_encoder_dilation = feature_encoder_dilation
        self.bottleneck_channels = bottleneck_channels
        self.use_high_pass = use_high_pass
        
        self.high_res_compressor = nn.Conv2d(high_res_channels, self.bottleneck_channels, 1)
        self.low_res_compressor = nn.Conv2d(low_res_channels, self.bottleneck_channels, 1)
        
        self.low_pass_generator = nn.Conv2d(
            self.bottleneck_channels,
            low_freq_kernel_size ** 2 * self.upsampling_groups * self.scale_ratio * self.scale_ratio,
            self.feature_encoder_kernel,
            padding=int((self.feature_encoder_kernel - 1) * self.feature_encoder_dilation / 2),
            dilation=self.feature_encoder_dilation,
            groups=1)
        
        self.high_pass_generator = nn.Conv2d(
            self.bottleneck_channels,
            high_freq_kernel_size ** 2 * self.upsampling_groups * self.scale_ratio * self.scale_ratio,
            self.feature_encoder_kernel,
            padding=int((self.feature_encoder_kernel - 1) * self.feature_encoder_dilation / 2),
            dilation=self.feature_encoder_dilation,
            groups=1)
        
        low_pass_padding = 0
        high_pass_padding = 0
        
        self.register_buffer('low_pass_window', 
                           torch.FloatTensor(create_2d_window(low_freq_kernel_size + 2 * low_pass_padding, 
                                                             low_freq_kernel_size + 2 * low_pass_padding))[None, None,])
        self.register_buffer('high_pass_window', 
                           torch.FloatTensor(create_2d_window(high_freq_kernel_size + 2 * high_pass_padding, 
                                                             high_freq_kernel_size + 2 * high_pass_padding))[None, None,])
        
        self.corner_alignment = corner_alignment
        self.interpolation_mode = interpolation_mode
        
        self._initialize_parameters()

    def _initialize_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                xavier_init(module, distribution='uniform')
        
        initialize_weights(self.low_pass_generator, std_val=0.001)
        if self.use_high_pass:
            initialize_weights(self.high_pass_generator, std_val=0.001)

    def normalize_filter_kernel(self, filter_mask, kernel_dim, scale_factor=None, window_weight=1):
        if scale_factor is not None:
            filter_mask = F.pixel_shuffle(filter_mask, self.scale_ratio)
        
        batch_size, mask_channels, height, width = filter_mask.size()
        channels_per_kernel = int(mask_channels / float(kernel_dim**2))
        
        filter_mask = filter_mask.view(batch_size, channels_per_kernel, -1, height, width)
        filter_mask = F.softmax(filter_mask, dim=2, dtype=filter_mask.dtype)
        filter_mask = filter_mask.view(batch_size, channels_per_kernel, kernel_dim, kernel_dim, height, width)
        
        filter_mask = filter_mask.permute(0, 1, 4, 5, 2, 3).view(batch_size, -1, kernel_dim, kernel_dim)
        
        filter_mask = filter_mask * window_weight
        filter_mask /= filter_mask.sum(dim=(-1, -2), keepdims=True)
        
        filter_mask = filter_mask.view(batch_size, channels_per_kernel, height, width, -1)
        filter_mask = filter_mask.permute(0, 1, 4, 2, 3).view(batch_size, -1, height, width).contiguous()
        
        return filter_mask

    def forward(self, high_res_features, low_res_features, enable_checkpoint=False):
        """前向传播"""
        if enable_checkpoint:
            return checkpoint(self._process_features, high_res_features, low_res_features)
        else:
            return self._process_features(high_res_features, low_res_features)

    def _process_features(self, high_res_features, low_res_features):

        compressed_high_res = self.high_res_compressor(high_res_features)
        compressed_low_res = self.low_res_compressor(low_res_features)

        high_freq_mask_hr = self.high_pass_generator(compressed_high_res)
        normalized_high_freq_mask = self.normalize_filter_kernel(
            high_freq_mask_hr, self.high_freq_kernel_size, window_weight=self.high_pass_window)
        
        high_freq_enhanced = compressed_high_res + compressed_high_res - carafe(
            compressed_high_res, normalized_high_freq_mask, self.high_freq_kernel_size, 
            self.upsampling_groups, 1)
        
        low_freq_mask_hr = self.low_pass_generator(high_freq_enhanced)
        normalized_low_freq_mask = self.normalize_filter_kernel(
            low_freq_mask_hr, self.low_freq_kernel_size, window_weight=self.low_pass_window)
        
        low_freq_mask_lr_raw = self.low_pass_generator(compressed_low_res)
        upsampled_low_freq_mask = F.interpolate(
            carafe(low_freq_mask_lr_raw, normalized_low_freq_mask, 
                   self.low_freq_kernel_size, self.upsampling_groups, 2), 
            size=compressed_high_res.shape[-2:], 
            mode=self.interpolation_mode)
        
        combined_low_freq_mask = low_freq_mask_hr + upsampled_low_freq_mask
        final_low_freq_mask = self.normalize_filter_kernel(
            combined_low_freq_mask, self.low_freq_kernel_size, window_weight=self.low_pass_window)
        
        high_freq_mask_lr_raw = self.high_pass_generator(compressed_low_res)
        upsampled_high_freq_mask = F.interpolate(
            carafe(high_freq_mask_lr_raw, final_low_freq_mask, 
                   self.low_freq_kernel_size, self.upsampling_groups, 2), 
            size=compressed_high_res.shape[-2:], 
            mode=self.interpolation_mode)
        
        combined_high_freq_mask = high_freq_mask_hr + upsampled_high_freq_mask
        final_high_freq_mask = self.normalize_filter_kernel(
            combined_high_freq_mask, self.high_freq_kernel_size, window_weight=self.high_pass_window)

        upsampled_low_res = carafe(low_res_features, final_low_freq_mask, 
                                  self.low_freq_kernel_size, self.upsampling_groups, 2)

        high_freq_component = high_res_features - carafe(
            high_res_features, final_high_freq_mask, self.high_freq_kernel_size, 
            self.upsampling_groups, 1)

        enhanced_high_res = high_freq_component + high_res_features
     
        return final_low_freq_mask, enhanced_high_res, upsampled_low_res
