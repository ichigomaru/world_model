import os

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(
        self,
        action: np.ndarray,
        images: np.ndarray,
        joint: np.ndarray,
        n_obs_steps: int,  # 観測ステップ数
        horizon: int,
    ):
        # 画像は0-255の整数値を0-1のfloatに変換
        self.images = torch.from_numpy(images).float() / 255.0
        self.action = torch.from_numpy(action).float()
        self.joint = torch.from_numpy(joint).float()
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon

        stats_path = "conf/data_info.yaml"
        print(f"正規化のための統計情報を '{stats_path}' から読み込んでいます...")
        try:
            with open(stats_path, "r") as f:
                norm_stats = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"エラー: '{stats_path}' が見つかりません。")
            print("事前に `generate_data_info_script` を実行して、統計情報ファイルを作成してください。")
            raise

        # --- ステップ2: 読み込んだ値をPyTorchテンソルとして保持 ---
        self.action_min = torch.tensor(norm_stats["action_min"], dtype=torch.float32)
        self.action_max = torch.tensor(norm_stats["action_max"], dtype=torch.float32)
        self.follower_min = torch.tensor(norm_stats["follower_min"], dtype=torch.float32)
        self.follower_max = torch.tensor(norm_stats["follower_max"], dtype=torch.float32)

        # --- ステップ3: 読み込んだ統計情報を使ってデータを正規化 ---
        self.action = self.normalize(self.action, self.action_min, self.action_max)
        self.joint = self.normalize(self.joint, self.follower_min, self.follower_max)

        # --- [DEBUG] 正規化後の最初のデータを確認 ---
        # print("\n--- [DEBUG] 正規化後の最初のデータ ---")
        # if len(self.action) > 0:
        #     print(f"最初のActionデータ (正規化後): {self.action[0]}")
        # if len(self.joint) > 0:
        #     print(f"最初のFollowerデータ (正規化後): {self.joint[0]}")
        # print("-------------------------------------\n")



    def normalize(self, x, min_val, max_val):
        epsilon = 1e-8
        return (x - min_val) / (max_val - min_val + epsilon) * 2 - 1

    def denormalize(self, x, min_val, max_val):
        return (x + 1) / 2 * (max_val - min_val) + min_val

    def denormalize_action(self, x):
        return self.denormalize(x, self.action_min, self.action_max)

    def __len__(self):
        return len(self.action) - (self.n_obs_steps + self.horizon) + 1

    def __getitem__(self, idx):
        # idx は時間ステップの開始位置（t0）
        act = self.action[
            idx + self.n_obs_steps : idx + self.n_obs_steps + self.horizon
        ]  # [horizon, action_dim]
        return {
            "observation.image": self.images[idx : idx + self.n_obs_steps],  # n_obs_steps分の画像系列
            "observation.state": self.joint[idx : idx + self.n_obs_steps],  # n_obs_steps分の状態系列
            "action": act,
            "action_is_pad": torch.tensor(
                [False] * min(self.horizon, len(self.action) - (idx + self.n_obs_steps))  # Falseが実データ
                + [True]
                * max(
                    0, self.horizon - (len(self.action) - (idx + self.n_obs_steps))
                ),  # Trueがパディングでダミー
                dtype=torch.bool,
            ),
        }
