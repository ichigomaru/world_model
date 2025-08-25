#Vision Network : CNN

import torch
import torch.nn as nn
import numpy as np

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
        obs_size = [48, 64]

        #エンコーダー
        self.encoder = self._build_conv_layers(
            channels, kernels, strides, paddings,
        )

        #MLP
        self.fc = _build_mlp(
            get_conved_size(
                obs_size,
                channels,
                kernels,
                strides,
                paddings,
            ), #flattenされた特徴量の数
            latent_obs_dim,   #研修2で*2してるのってなんでだっけ？-> 潜在分布を表現するために平均と分散の両方を出力するための次元数を意味
            mlp_hidden_dim,
            n_mlp_layers,
        )

    #順伝播=NN
    def forward(self, x):
        x = self.encoder(x) #エンコーダー
        #print("after encoder, x.shape:", x.shape) 
        x = torch.flatten(x, start_dim=1) #flatten
        x = self.fc(x) #MLP層
        return x #最終的にこれを返す

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



def _build_mlp(input_dim, output_dim, hidden_dim, n_layers):
    layers = []
    layers.append(nn.Flatten()) #フラット化

    if n_layers == 0:
        layers.append(nn.Linear(input_dim, output_dim)) #隠れ層がない時に単純に入力層から出力層への全結合層を作ります。

    else:
        layers.append(nn.Linear(input_dim, hidden_dim)) 
        layers.append(nn.ReLU())
        for _ in range(n_layers - 1): #隠れ層数に基づいて、追加の隠れ層を構築
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim)) 
        
    return nn.Sequential(*layers) #上で作ったすべてのレイヤーを順番に並べたネットワークを制作する

# 画像のサイズを計算する関数
def get_conved_size(
    obs_shape=None, 
    channels=None, 
    kernels=None, 
    strides=None, 
    paddings=None,
    ):
    
    conved_shape = obs_shape
    for i in range(len(channels)):
        conved_shape = conv_out_shape(
            conved_shape,
            paddings[i],  #[1, 1]
            kernels[i],  #[3, 3]
            strides[i],  #[2, 2]
        )
        conved_size = channels[-1] * np.prod(conved_shape).item() #np.prod().item -> 特徴マップの面積
    return conved_size
    #フラット化後の特徴量の数
        
def conv_out_shape(h_in, padding, kernel_size, stride):
    return tuple(conv_out(x, padding, kernel_size, stride) for x in h_in)

def conv_out(h_in, padding, kernel_size, stride):
    return int((h_in + 2.0 * padding - (kernel_size - 1.0) - 1.0) / stride + 1.0)
