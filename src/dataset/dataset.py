import numpy as np
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self,
                action,
                images,
                joint,
                sequence_length: int):
        
        self.action = torch.from_numpy(action).float()
        self.images = torch.from_numpy(images).float()
        self.joint = torch.from_numpy(joint).float()
        self.sequence_length = sequence_length
        # 1. NumPy配列をTensorに変換
        images_tensor = torch.from_numpy(images).float()
        
        # 2. ピクセル値を [0, 255] から [-1, 1] の範囲に正規化
        self.images = (images_tensor / 127.5) - 1.0
        
    def __len__(self):
        return len(self.action) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        start = idx
        end = start + self.sequence_length

        return {
            "images": self.images[start:end],      # <--- 変更点: 1枚ではなくシーケンスを返す
            "joints": self.joint[start:end],       # <--- 変更点
            "actions": self.action[start:end], 
        }

'''
出力
"images": (バッチサイズ, 10, 3, 240, 320) の形状を持つ、10フレーム分の画像シーケンス
"joints": (バッチサイズ, 10, 6) の形状を持つ、10タイムステップ分の関節角度シーケンス
"actions": (バッチサイズ, 10, 6) の形状を持つ、10タイムステップ分のアクションシーケンス
'''