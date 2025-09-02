import torch
import torch.nn as nn
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

def get_conved_size(obs_shape, channels, kernels, strides, paddings):
    conved_shape = obs_shape
    for i in range(len(channels)):
        conved_shape = conv_out_shape(conved_shape, paddings[i], kernels[i], strides[i])
    conved_size = channels[-1] * np.prod(conved_shape).item()
    return int(conved_size)


class VisionEncoder(nn.Module):
    def __init__(
        self,
        channels,
        kernels,
        strides,
        paddings,
        latent_obs_dim, #潜在変数の次元
        mlp_hidden_dim,
        n_mlp_layers):
        super().__init__()

        #画像サイズ
        obs_size = [240, 320]

        #エンコーダー
        self.encoder = self._build_conv_layers(
            channels, kernels, strides, paddings,
        )

        conved_size = get_conved_size(
            obs_size, channels, kernels, strides, paddings
        )

        self.fc = _build_mlp(
            conved_size,
            latent_obs_dim, 
            mlp_hidden_dim,
            n_mlp_layers,
        )

    #順伝播=NN
    def forward(self, x):
        
        x = self.encoder(x) #エンコーダー
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x) #MLP層
        return x 
    

    def _build_conv_layers(self, channels, kernels, strides, paddings):
        layers = []

        for i in range(len(channels)):
            layers.append(
                nn.Conv2d(
                    in_channels=channels[i-1] if i > 0 else 3,  # RGB3
                    out_channels=channels[i], #8
                    kernel_size=kernels[i], #3
                    stride=strides[i], #2
                    padding=paddings[i], #1  3 -> 8 チャンネルへ
                )
            )
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)
        #＊はリストを展開して渡す



