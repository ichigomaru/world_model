import os

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import yaml
from box import Box
from safetensors.torch import load_file

from model.vision_RNN import VisionEncoder

# 設定ファイル読み込み
with open("./conf/conf.yaml", "r") as yml:
    cfg = Box(yaml.safe_load(yml))

device = torch.device(cfg.wandb.config.device if torch.cuda.is_available() else "cpu")

# Vision モデル構築
vision = VisionEncoder(
    channels=cfg.model.vision.channels,
    kernels=cfg.model.vision.kernels,
    strides=cfg.model.vision.strides,
    paddings=cfg.model.vision.paddings,
    latent_obs_dim=cfg.model.latent_obs_dim,
    mlp_hidden_dim=cfg.model.mlp.mlp_hidden_dim,
    n_mlp_layers=cfg.model.mlp.n_mlp_layers,
).to(device)

# 学習済みモデル読み込み
best_epoch_path = f"result/{cfg.wandb.train_name}/model/best_epoch.yaml"
if not os.path.exists(best_epoch_path):
    raise FileNotFoundError(f"{best_epoch_path} が見つかりません")
with open(best_epoch_path, "r") as f:
    best_epoch = yaml.safe_load(f)["best_epoch"]

model_path = f"result/{cfg.wandb.train_name}/model/vision_epoch_{best_epoch}.safetensors"
vision.load_state_dict(load_file(model_path))
vision.eval()

# 学習時の画像を読み込み
image_tensor = torch.load(f"output/{cfg.wandb.train_name.replace(':', '_')}/robot_test/raw/images.pt")
image_tensor = image_tensor.float() / 255.0  # 正規化
image_tensor = image_tensor.to(device)

# 画像サイズをモデルの入力に合わせてリサイズ

resize = T.Resize((48, 64))
image_tensor = resize(image_tensor)

# モデルを通して潜在変数取得
with torch.no_grad():
    latents = vision(image_tensor)

latents_np = latents.cpu().numpy()

# 可視化: 各次元の min/max を表示
latent_min = latents_np.min(axis=0)
latent_max = latents_np.max(axis=0)

plt.figure(figsize=(10, 5))
plt.plot(latent_min, label="min")
plt.plot(latent_max, label="max")
plt.title("Latent Feature Range")
plt.xlabel("Latent Dimension")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.savefig("output/latent_range.png")
print("Saved latent range visualization to output/latent_range.png")
