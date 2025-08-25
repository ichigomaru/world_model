import os

import blosc2
import numpy as np
import yaml


def load_blosc2(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        return blosc2.unpack_array2(f.read())


"""
r読み取り。bバイナリ(圧縮ファイルなど)
blosc2.unpack_array2(...)はblosc2が提供しているライブラリで、圧縮されたデータの復元
"""

# 読み込み
action_states = load_blosc2("data/right/action_states.blosc2")
image_states = load_blosc2("data/right/images_states.blosc2")
joint_states = load_blosc2("data/right/joint_states.blosc2")

# 中身確認
print("action_states shape:", action_states.shape)
print("image_states shape:", image_states.shape)
print("joint_states shape:", joint_states.shape)

# print(image_states[10])
# 2-0のデータを確認。50シークエンス、6個の関節の角度のデータ

# 各ファイルの次元ごとの最大値と最小値を取得。yamlに保存

# スケール前のmin/maxを各次元ごとに取得
action_min = action_states.min(axis=(0, 1)).tolist()  # shape: (6,)
action_max = action_states.max(axis=(0, 1)).tolist()

follower_min = joint_states.min(axis=(0, 1)).tolist()  # shape: (6,)
follower_max = joint_states.max(axis=(0, 1)).tolist()

# 保存用の辞書を構築
data_info = {
    "action_min": action_min,
    "action_max": action_max,
    "follower_min": follower_min,
    "follower_max": follower_max,
}

save_path = os.path.join("conf", "data_info.yaml")
os.makedirs("conf", exist_ok=True)

with open(save_path, "w") as f:
    yaml.dump(data_info, f, default_flow_style=None)

print("min/max save OK!")

# print("action_states min:", action_states_min)
# print("action_states max:", action_states_max)
# print("image_states min:", image_states_min)
# print("image_states max:", image_states_max)
# print("joint_states min:", joint_states_min)
# print("joint_states max:", joint_states_max)


# scale_to_range関数を定義
def scale_to_range(data, min_val, max_val, target_min=-1.0, target_max=1.0):
    return target_min + (data - min_val) * (target_max - target_min) / (max_val - min_val)


# スケーリングされたデータを保存する配列を用意
action_states_scaled = np.zeros_like(action_states, dtype=np.float32)

# 各次元の最小値と最大値を取得
for dim in range(6):
    values = action_states[:, :, dim]
    max_val = values.max()
    min_val = values.min()

    scaled = scale_to_range(values, min_val, max_val)
    action_states_scaled[:, :, dim] = scaled

print("スケーリング後の3次元の最大値:", action_states_scaled[:, :, 3].max())
print("スケーリング後の3次元の最小値:", action_states_scaled[:, :, 3].min())


# データの形状: (シーケンス数, フレーム数, 次元数)
num_sequences, seq_len, dim = action_states_scaled.shape

"""
# グラフの描画
fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True)

for dim_idx in range(6):
    for seq_idx in range(num_sequences):
        axes[dim_idx].plot(
            range(seq_len),
            action_states_scaled[seq_idx, :, dim_idx],
            label=f"Seq{seq_idx}" if dim_idx == 0 else "",
            alpha=0.5,
        )

    axes[dim_idx].set_ylabel(f"Dim {dim_idx+1}", fontsize=10)
    axes[dim_idx].grid(True)

# X軸のラベルと目盛りを詳細に設定
axes[-1].set_xlabel("Frame Index", fontsize=12)

# タイトルの設定
plt.suptitle(f"Action States (Sequence all)", fontsize=14, y=1.02)

plt.tight_layout()
plt.show()
"""

# image_statesのスケーリング（1/255に正規化）
image_states_scaled = image_states.astype(np.float32) / 255.0

# print("スケーリング後のimage_statesの最大値:", image_states_scaled.max())
# print("スケーリング後のimage_statesの最小値:", image_states_scaled.min())

# print(image_states_scaled)

# joint_statesのスケーリング

# スケーリングされたデータを保存する配列を用意
joint_states_scaled = np.zeros_like(joint_states, dtype=np.float32)

# 各次元の最小値と最大値を取得
for dim in range(6):
    values = joint_states[:, :, dim]
    max_val = values.max()
    min_val = values.min()

    scaled = scale_to_range(values, min_val, max_val)
    joint_states_scaled[:, :, dim] = scaled


# データの形状: (シーケンス数, フレーム数, 次元数)
num_sequences, seq_len, dim = joint_states_scaled.shape

"""
# グラフの描画
fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True)

for dim_idx in range(6):
    for seq_idx in range(num_sequences):
        axes[dim_idx].plot(
            range(seq_len),
            joint_states_scaled[seq_idx, :, dim_idx],
            label=f"Seq{seq_idx}" if dim_idx == 0 else "",
            alpha=0.5,
        )

    axes[dim_idx].set_ylabel(f"Dim {dim_idx+1}", fontsize=10)
    axes[dim_idx].grid(True)

# X軸のラベルと目盛りを詳細に設定
axes[-1].set_xlabel("Frame Index", fontsize=12)

# タイトルの設定
plt.suptitle(f"Joint States (Sequence all)", fontsize=14, y=1.02)

plt.tight_layout()
plt.show()
"""


# 保存先パス
np.save("data/action_states_scaled.npy", action_states_scaled)
np.save("data/image_states_scaled.npy", image_states_scaled)
np.save("data/joint_states_scaled.npy", joint_states_scaled)

print("スケーリング後のデータを保存しました。")

# print(action_states_scaled)
# print(image_states_scaled)
print(joint_states_scaled)
