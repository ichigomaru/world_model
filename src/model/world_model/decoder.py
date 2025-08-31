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

def get_encoder_feature_sizes(obs_shape, channels, kernels, strides, paddings):
    """エンコーダーの各層の出力サイズを計算する"""
    sizes = []
    h, w = obs_shape
    sizes.append((h, w)) # 初期サイズを追加
    for i in range(len(channels)):
        h = conv_out(h, paddings[i], kernels[i], strides[i])
        w = conv_out(w, paddings[i], kernels[i], strides[i])
        sizes.append((h, w))
    return sizes

def calculate_output_padding(h_in, w_in, h_out, w_out, stride, kernel, padding):
    """必要なoutput_paddingを計算する"""
    h_pad = h_out - ((h_in - 1) * stride - 2 * padding + kernel)
    w_pad = w_out - ((w_in - 1) * stride - 2 * padding + kernel)
    return h_pad, w_pad

class VisionDecoder(nn.Module):
    def __init__(
        self,
        channels,
        kernels,
        strides,
        paddings,
        latent_obs_dim,
        mlp_hidden_dim,
        n_mlp_layers
    ):
        super().__init__()
        
        # 1. エンコーダーの各層の特徴マップサイズを事前に計算
        self.encoder_feature_sizes = get_encoder_feature_sizes(
            [240, 320], channels, kernels, strides, paddings
        )
        
        # エンコーダーの最終出力形状を取得
        final_h, final_w = self.encoder_feature_sizes[-1]
        self.pre_flatten_shape = (channels[-1], final_h, final_w)
        flattened_size = np.prod(self.pre_flatten_shape).item()

        # 2. 逆MLP
        self.fc = _build_mlp(
            latent_obs_dim,
            flattened_size,
            mlp_hidden_dim,
            n_mlp_layers
        )

        # 3. 逆CNN
        self.decoder = self._build_deconv_layers(channels, kernels, strides, paddings)

    def _build_deconv_layers(self, channels, kernels, strides, paddings):
        # パラメータを逆順にする
        rev_channels = list(reversed(channels))
        rev_kernels = list(reversed(kernels))
        rev_strides = list(reversed(strides))
        rev_paddings = list(reversed(paddings))
        
        # エンコーダーのサイズリストも逆順に
        rev_sizes = list(reversed(self.encoder_feature_sizes))

        layers = []
        for i in range(len(rev_channels)):
            in_channels = rev_channels[i]
            out_channels = rev_channels[i+1] if i < len(rev_channels) - 1 else 3
            
            # 復元元と復元先のサイズを取得
            h_in, w_in = rev_sizes[i]
            h_out, w_out = rev_sizes[i+1]

            # 適切なoutput_paddingを計算
            output_padding = calculate_output_padding(
                h_in, w_in, h_out, w_out, rev_strides[i], rev_kernels[i], rev_paddings[i]
            )
            
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=rev_kernels[i],
                    stride=rev_strides[i],
                    padding=rev_paddings[i],
                    output_padding=output_padding
                )
            )
            if i < len(rev_channels) - 1:
                layers.append(nn.ReLU())

        layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, *self.pre_flatten_shape)
        x = self.decoder(x)
        
        # interpolateに頼る必要がなくなる！
        # assert x.shape[-2:] == (240, 320) # 念のため確認
            
        return x