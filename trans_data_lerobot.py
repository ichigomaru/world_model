import os
import torch
import numpy as np
import blosc2
from datasets import Dataset, Features, Value, Array3D, Sequence
from huggingface_hub import HfApi

# blosc2読み込み関数は不要なので削除

# numpy読み込みに変更
joint = np.load("data/merged/joint_states.npy")
action = np.load("data/merged/action_states.npy")
image = np.load("data/merged/images_states.npy")

print(image.shape)

N, T = action.shape[:2] #80データの番号と50フレームのカウント

# flatten して1サンプルずつに展開
data = {
    "observation.image": [],
    "observation.state": [],
    "action": [],
    "episode_index": [],
    "frame_index": [],
}

for epi in range(N):
    for t in range(T):
        data["observation.image"].append(image[epi, t])
        data["observation.state"].append(joint[epi, t])
        data["action"].append(action[epi, t])
        data["episode_index"].append(epi) #80データあるうちの何番目か
        data["frame_index"].append(t) #50シークエンスあるうちの何番目か

# Dataset作成
features = Features({
    "observation.image": Array3D(dtype="uint8", shape=(3, 240, 320)),
    "observation.state": Sequence(feature=Value("float32"), length=6),
    "action": Sequence(feature=Value("float32"), length=6),
    "episode_index": Value("int32"),
    "frame_index": Value("int32"),
})
dataset = Dataset.from_dict(data, features=features)

# 保存（HuggingFace hub に push するなら .push_to_hub）
dataset.save_to_disk("lerobot/data_merged")  # ローカルに保存

# 最初のサンプルを取得して表示
sample = dataset[0]
print("Sample 0:")
for key, value in sample.items():
    print(f"{key}: data={np.array(value)}, type={type(value)}")