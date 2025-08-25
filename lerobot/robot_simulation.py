import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torchvision.transforms as T
import yaml
from box import Box
from datasets import load_from_disk
from safetensors.torch import load_file

from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionModel, DiffusionPolicy
from lerobot.configs.types import FeatureType, PolicyFeature
from src.model.vision import VisionEncoder

with open("./conf/conf.yaml", "r") as yml:
    cfg = Box(yaml.safe_load(yml))

device = torch.device(cfg.wandb.config.device if torch.cuda.is_available() else "cpu")

episode_index = 60  # ← 任意のエピソード番号に変更可能

# モデル構築
vision = VisionEncoder(
    channels=cfg.model.vision.channels,
    kernels=cfg.model.vision.kernels,
    strides=cfg.model.vision.strides,
    paddings=cfg.model.vision.paddings,
    latent_obs_dim=cfg.model.latent_obs_dim,
    mlp_hidden_dim=cfg.model.mlp.mlp_hidden_dim,
    n_mlp_layers=cfg.model.mlp.n_mlp_layers,
).to(device)

# diffusion pollicy
config = DiffusionConfig(
    n_obs_steps=cfg.model.diffusion.n_obs_steps,
    horizon=cfg.model.diffusion.horizon,
    n_action_steps=cfg.model.diffusion.n_action_steps,
    vision_backbone="resnet18",
    crop_shape=(240, 320),
    input_features={
        "observation.image": PolicyFeature(FeatureType.VISUAL, shape=(3, 240, 320)),
        "observation.state": PolicyFeature(FeatureType.STATE, shape=(6,)),
    },
    output_features={"action": PolicyFeature(FeatureType.ACTION, shape=(6,))},
)

diffusion_model = DiffusionModel(config).to(device)
policy = DiffusionPolicy(config, vision_encoder=vision, diffusion_model=diffusion_model).to(device)


# ベストモデルの読み込み
best_epoch_path = f"result/{cfg.wandb.train_name}/model/best_epoch.yaml"
if os.path.exists(best_epoch_path):
    with open(best_epoch_path, "r") as f:
        best_epoch_data = yaml.safe_load(f)
    best_epoch = best_epoch_data["best_epoch"]

    policy_path = f"result/{cfg.wandb.train_name}/model/policy_epoch_{best_epoch}.safetensors"

    policy.load_state_dict(load_file(policy_path))
else:
    raise FileNotFoundError(f"{best_epoch_path} が存在しない。トレーニングでbest_epoch.yamlを保存したか確認")


# 正規化準備
dataset = load_from_disk("data_merged")
# action の max/min をデータセットから直接計算する
all_actions_tensor = torch.tensor(dataset["action"])
min_vals_action = torch.min(all_actions_tensor, dim=0)[0].to(device)
max_vals_action = torch.max(all_actions_tensor, dim=0)[0].to(device)

# joint の max/min も同様に計算
all_joints_tensor = torch.tensor(dataset["observation.state"])
min_vals_joint = torch.min(all_joints_tensor, dim=0)[0].to(device)
max_vals_joint = torch.max(all_joints_tensor, dim=0)[0].to(device)

# 正規化関数スケーリング
def normalize(x, min_val, max_val):
    x = torch.clamp(x, min=min_val, max=max_val)  # 外れ値をカット
    return (x - min_val) / (max_val - min_val) * 2 - 1


# 逆正規化
def denormalize(x, min_val, max_val):
    return ((x + 1) / 2) * (max_val - min_val) + min_val


# 推論ループ
vision.eval()
policy.eval()

images = []
joints = []
image_features = []

NUM_STEPS = cfg.run.steps
FPS = cfg.run.fps
n_obs_steps = cfg.model.diffusion.n_obs_steps
horizon = cfg.model.diffusion.horizon
n_action_steps = cfg.model.diffusion.n_action_steps


# ＝＝＝推論に使うためのデータセットの読み込み＝＝＝
dataset = load_from_disk("data_merged")

# 最初のデータのn_obs_steps個の観測を取得
episode_n = dataset.filter(
    lambda x: x["episode_index"] == episode_index
)  # 情報選別、フィルタリング。lambdaにある条件のみ通す。
episode_n = sorted(episode_n, key=lambda x: x["frame_index"])

episode_n_all = dataset.filter(lambda x: x["episode_index"] == episode_index)
episode_n_all = sorted(episode_n_all, key=lambda x: x["frame_index"])


action_predictions = []

t = 0
while t + n_obs_steps + n_action_steps <= len(episode_n) and len(action_predictions) < NUM_STEPS:
    images = [torch.tensor(episode_n[t + i]["observation.image"]).float() / 255.0 for i in range(n_obs_steps)]
    states = [
        normalize(torch.tensor(episode_n[t + i]["observation.state"]), min_vals_joint, max_vals_joint)
        for i in range(n_obs_steps)
    ]

    batch = {
        "observation.images": torch.stack(images).unsqueeze(0),  # (1, n_obs_steps, 3, 240, 320)
        "observation.state": torch.stack(states).unsqueeze(0),  # (1, n_obs_steps, 6)
    }

    with torch.no_grad():
        images_tensor = torch.stack(images)  # (n_obs_steps, 3, 240, 320)
        images_tensor = images_tensor.unsqueeze(0).to(device)  # (1, n_obs_steps, 3, 240, 320)
        B, T, C, H, W = images_tensor.shape  # noqa: N806
        images_flat = images_tensor.reshape(B * T, C, H, W)

        features_flat = vision(images_flat)  # (B*T, feature_dim)
        image_features = features_flat.reshape(B, T, -1)  # (1, n_obs_steps, feature_dim)
        batch["image_features"] = image_features  # ←ここが重要！
        pred_actions = policy.diffusion.generate_actions(batch)[0]

    for i in range(n_action_steps):
        if len(action_predictions) >= NUM_STEPS:
            break
        action_predictions.append(pred_actions[i])

    t += n_action_steps
# action_predictionsに全timestep分のアクションの関節角度が保存される

# --- 準備 ---
# 保存用ディレクトリ
train_name_safe = cfg.wandb.train_name.replace(":", "_")
save_root = f"sim_result/{train_name_safe}"
os.makedirs(save_root, exist_ok=True)

# 予測データ（正規化済み）
normalized_predictions_tensor = torch.stack(action_predictions)

# 正解データ（生データと正規化済みデータの両方を準備）
ground_truth_raw_tensor = torch.stack([torch.tensor(x["action"]) for x in episode_n_all[:NUM_STEPS]])
ground_truth_normalized_tensor = normalize(ground_truth_raw_tensor.to(device), min_vals_action, max_vals_action)

# --- プロット1: 正規化済みデータ (-1 ~ 1の範囲) ---
print("プロット1: 正規化済みデータを可視化中...")
fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
for i in range(6):
    # Paddingを追加して長さを合わせる
    padded_preds = torch.cat([torch.full((n_obs_steps,), float("nan")), normalized_predictions_tensor[:, i].cpu()])
    
    axes[i].plot(padded_preds, label="action Prediction", linestyle="--")
    axes[i].plot(ground_truth_normalized_tensor[:, i].cpu(), label="action data", linestyle="-")
    axes[i].set_ylabel(f"Joint {i + 1}")
    axes[i].set_ylim(-1.1, 1.1)  # Y軸を-1.1から1.1に固定
    axes[i].grid(True, linestyle=':', alpha=0.6)
    if i == 0:
        axes[i].legend(loc="upper right")

axes[-1].set_xlabel("Time step")
axes[-1].xaxis.set_major_locator(ticker.MultipleLocator(10))
axes[-1].xaxis.set_minor_locator(ticker.MultipleLocator(1))

plt.suptitle(f"action prediction vs action data (normarized) (Episode {episode_index})")
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
prediction_plot_path_norm = os.path.join(save_root, f"prediction_vs_action_normalized_e{episode_index}.png")
plt.savefig(prediction_plot_path_norm)
plt.close()
print(f"Saved normalized plot: {prediction_plot_path_norm}")


# --- プロット2: 逆正規化後のデータ (元のスケール) ---
print("\nプロット2: 逆正規化済みデータを可視化中...")
# 予測データを逆正規化
denormalized_predictions_tensor = denormalize(normalized_predictions_tensor, min_vals_action, max_vals_action)

fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
for i in range(6):
    # Paddingを追加して長さを合わせる
    padded_preds = torch.cat([torch.full((n_obs_steps,), float("nan")), denormalized_predictions_tensor[:, i].cpu()])

    axes[i].plot(padded_preds, label="action prediction", linestyle="--")
    axes[i].plot(ground_truth_raw_tensor[:, i], label="action data", linestyle="-")
    axes[i].set_ylabel(f"Joint {i + 1}")
    if i == 0:
        axes[i].legend(loc="upper right")

axes[-1].set_xlabel("Time step")
axes[-1].xaxis.set_major_locator(ticker.MultipleLocator(10))
axes[-1].xaxis.set_minor_locator(ticker.MultipleLocator(1))

plt.suptitle(f"action prediction vs action data(Episode {episode_index})")
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
prediction_plot_path_denorm = os.path.join(save_root, f"prediction_vs_action_denormalized_e{episode_index}.png")
plt.savefig(prediction_plot_path_denorm)
plt.close()
print(f"Saved de-normalized plot: {prediction_plot_path_denorm}")