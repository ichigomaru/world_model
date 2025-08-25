import os

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

# from src.utils.transform import ObsTransform


class MyDataset(Dataset):
    def __init__(
        self,
        action: np.ndarray,
        images: np.ndarray,
        joint: np.ndarray,
        # is_train: bool = True,
    ):
        self.action = torch.from_numpy(action).float()
        self.images = (
            torch.from_numpy(images).float() / 255.0
        )  # 画像データは0-255の範囲なので、0-1にスケールする
        self.joint = torch.from_numpy(joint).float()
        # self.is_train = is_train
        # self.images = ObsTransform(48, 64, is_train)(images)

        # スケール前のmin/maxを各次元ごとに取得
        self.action_min = self.action.amin(dim=(0, 1)).tolist()
        self.action_max = self.action.amax(dim=(0, 1)).tolist()
        follower_min = self.joint.amin(dim=(0, 1)).tolist()
        follower_max = self.joint.amax(dim=(0, 1)).tolist()

        # 保存用の辞書を構築
        data_info = {
            "action_min": self.action_min,
            "action_max": self.action_max,
            "follower_min": follower_min,
            "follower_max": follower_max,
        }

        save_path = os.path.join("conf", "data_info.yaml")
        os.makedirs("conf", exist_ok=True)

        with open(save_path, "w") as f:
            yaml.dump(data_info, f, default_flow_style=None)

        # tensorに変換
        self.action_min = torch.tensor(self.action_min).float()
        self.action_max = torch.tensor(self.action_max).float()
        follower_min = torch.tensor(follower_min).float()
        follower_max = torch.tensor(follower_max).float()

        # 正規化のためのパスを読み込み
        self.action = self.normalize(self.action, self.action_min, self.action_max)  # 正規化
        self.joint = self.normalize(self.joint, follower_min, follower_max)

    def normalize(self, x, min_val, max_val):
        return (x - min_val) / (max_val - min_val) * 2 - 1

    def denormalize(self, x, min_val, max_val):
        return (x + 1) / 2 * (max_val - min_val) + min_val

    def denormalize_action(self, x):
        return self.denormalize(x, self.action_min, self.action_max)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.joint[idx], self.action[idx]
