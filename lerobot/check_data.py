import matplotlib.pyplot as plt
import torch
import yaml
from box import Box
import os
import matplotlib.ticker as ticker
from datasets import load_from_disk

with open("./conf/conf.yaml", "r") as yml:
    cfg = Box(yaml.safe_load(yml))

device = torch.device(cfg.wandb.config.device if torch.cuda.is_available() else "cpu")


def normalize(x, min_val, max_val):
    # ゼロ除算を避けるための小さな値
    epsilon = 1e-8
    return (x - min_val) / (max_val - min_val + epsilon) * 2 - 1

# ＝＝＝推論に使うためのデータセットの読み込み＝＝＝
dataset = load_from_disk("data_merged")


# まず、'action'列の全データを1つのTensorにまとめる
all_actions_tensor = torch.tensor(dataset["observation.state"])
# 各次元（各関節）ごとの最小値と最大値を取得
min_vals = torch.min(all_actions_tensor, dim=0)[0]
max_vals = torch.max(all_actions_tensor, dim=0)[0]
print("計算完了。")
print(f"Min values per joint: {min_vals.numpy()}")
print(f"Max values per joint: {max_vals.numpy()}")




import pyarrow.ipc

# .arrowファイルを指定
file_path = "data_merged/data-00000-of-00001.arrow" # ファイル名は環境に合わせてください

# Arrow IPCストリーミング形式のファイルを読み込む
with open(file_path, 'rb') as f:
    reader = pyarrow.ipc.open_stream(f)
    
    # 設計図（スキーマ）を表示する
    print("Arrow file schema:")
    print(reader.schema)




fig, axes = plt.subplots(6, 1, figsize=(12, 15), sharex=True)
print("全エピソードの正規化済みactionデータをプロット中...")

# --- ステップ3: 全エピソードをループしてプロット ---
num_episodes = len(dataset.unique("episode_index"))
for episode_index in range(num_episodes):
    # 特定のエピソードのデータを抽出
    episode_n_all = dataset.filter(lambda x: x["episode_index"] == episode_index, num_proc=4) # 高速化のため並列処理
    episode_n_all = sorted(episode_n_all, key=lambda x: x["frame_index"])

    # actionデータをTensorに変換
    ground_truth_tensor = torch.stack([torch.tensor(x["observation.state"]) for x in episode_n_all])
    
    # ★★★ ここで正規化を実行 ★★★
    normalized_tensor = normalize(ground_truth_tensor, min_vals, max_vals)
    
    ground_truth_np = normalized_tensor.numpy()

    # 各関節のデータをプロット
    for i in range(6):
        axes[i].plot(ground_truth_np[:, i], alpha=0.2, color='gray')

# --- ステップ4: グラフの体裁を整える ---
for i in range(6):
    axes[i].set_ylabel(f'Joint {i+1}')
    # ★★★ Y軸の範囲を-1.1から1.1に設定 ★★★
    axes[i].set_ylim(-1.1, 1.1)
    axes[i].axhline(0, color='r', linestyle='--', linewidth=0.8) # ゼロの基準線を追加
    axes[i].grid(True, linestyle=':', alpha=0.6)

axes[0].set_title("follower_all")
axes[-1].set_xlabel("Time step")
axes[-1].xaxis.set_major_locator(ticker.MultipleLocator(10))
axes[-1].xaxis.set_minor_locator(ticker.MultipleLocator(1))
axes[-1].tick_params(axis='x', which='major', length=10, labelsize=10)
axes[-1].tick_params(axis='x', which='minor', length=5)

plt.tight_layout(rect=[0, 0, 1, 0.97]) # タイトルと重ならないように調整

# --- ステップ5: グラフの保存 ---
# train_name_safe = cfg.wandb.train_name.replace(":", "_") # cfgオブジェクトが利用可能ならこの行を有効化
train_name_safe = "default_run" # cfgがない場合の仮の名前
save_root = f'sim_result/check_data'
os.makedirs(save_root, exist_ok=True)
plot_path = os.path.join(save_root, 'follower_all.png')
plt.savefig(plot_path)
plt.close()

print(f"\n正規化されたグラフを保存しました: {plot_path}")
