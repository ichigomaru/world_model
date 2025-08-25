import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def _build_mlp(input_dim, output_dim, hidden_dim, n_layers):

    layers = []
    if n_layers == 0:
        layers.append(nn.Linear(input_dim, output_dim))
    else:
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)

def conv_out(h_in, padding, kernel_size, stride):
    return int((h_in + 2.0 * padding - (kernel_size - 1.0) - 1.0) / stride + 1.0)

def conv_out_shape(h_in, padding, kernel_size, stride):
    return tuple(conv_out(x, padding, kernel_size, stride) for x in h_in)


class VisionDecoder(nn.Module):
    def __init__(
        self,
        channels,
        kernels,
        strides,
        paddings,
        latent_obs_dim, # RSSMのstochastic_sizeが入る (例: 30)
        mlp_hidden_dim,
        n_mlp_layers
    ):
        super().__init__()

        # エンコーダーのCNNが出力した特徴マップの形状を計算
        self.pre_flatten_shape = self._get_pre_flatten_shape(
            [240, 320], channels, kernels, strides, paddings
        )
        
        flattened_size = np.prod(self.pre_flatten_shape).item()

        # 1. 逆MLP: 潜在変数(例: 30)を、CNNの出力チャネル数(32)に変換する
        self.fc = _build_mlp(
            latent_obs_dim,
            flattened_size,
            mlp_hidden_dim,
            n_mlp_layers
        )

        # 2. 逆CNN: 特徴マップを元の画像サイズに復元するデコーダー層
        self.decoder = self._build_deconv_layers(channels, kernels, strides, paddings)

    def _get_pre_flatten_shape(self, obs_shape, channels, kernels, strides, paddings):
        conved_shape = list(obs_shape)
        for i in range(len(channels)):
            conved_shape = conv_out_shape(conved_shape, paddings[i], kernels[i], strides[i])
        return (channels[-1], *conved_shape)

    def _build_deconv_layers(self, channels, kernels, strides, paddings):
        rev_channels = list(reversed(channels))
        rev_kernels = list(reversed(kernels))
        rev_strides = list(reversed(strides))
        rev_paddings = list(reversed(paddings))

        layers = []
        for i in range(len(rev_channels)):
            in_channels = rev_channels[i]
            out_channels = rev_channels[i+1] if i < len(rev_channels) - 1 else 3
            
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=rev_kernels[i],
                    stride=rev_strides[i],
                    padding=rev_paddings[i],
                    output_padding=0 # interpolateを使うため、厳密な調整は不要
                )
            )
            if i < len(rev_channels) - 1:
                layers.append(nn.ReLU())

        layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (B, stochastic_size) の形状を持つ潜在変数のテンソル。

        Returns:
            torch.Tensor: (B, 3, 240, 320) の形状を持つ復元されたRGB画像のテンソル。
        """

        x = self.fc(x)
        x = x.view(-1, *self.pre_flatten_shape)
        x = self.decoder(x)

        if x.shape[-2:] != (240, 320):
            x = F.interpolate(x, size=(240, 320), mode='bilinear', align_corners=False)
            
        return x