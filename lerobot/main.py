import os
import random

import numpy as np
import torch
import torch.nn as nn
import wandb
import yaml
from box import Box
from datasets import load_from_disk
from safetensors.torch import load_file

from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import (
    DiffusionModel,
    DiffusionPolicy,
)
from lerobot.configs.types import FeatureType, PolicyFeature
from src.model.vision import VisionEncoder
from src.dataloader.dataloader import MyDataloader
from src.dataset.dataset import MyDataset
from src.trainer.trainer import TrainerSeparated
from src.utils.print_model import print_model_structure

device = torch.device("mps" if torch.cuda.is_available() else "cpu")

# シードの固定
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)

os.chdir(
    os.path.dirname(os.path.abspath(__file__))
)  # ch = change このスクリプトと同じディレクトリをカレントディレクトリにする __file__はこのスクリプトのパスを取得する つまりkensyu_3に移動
os.makedirs("output", exist_ok=True)

with open("conf/conf.yaml", "r") as yml:
    cfg = Box(yaml.safe_load(yml))

os.makedirs(f"result/{cfg.wandb.train_name}", exist_ok=True)
wandb.init(project=cfg.wandb.project_name, config=cfg.wandb.config, name=cfg.wandb.train_name)


# action = load_blosc2('data/merged/action_states.blosc2')
# images = load_blosc2('data/merged/image_states.blosc2')
# joint = load_blosc2('data/merged/joint_states.blosc2')

data = load_from_disk("data_merged")

# カラムごとに NumPy 配列として取り出す
images = np.stack(data["observation.image"])  # shape: (N, 3, 240, 320)
joint = np.stack(data["observation.state"])  # shape: (N, 6)
action = np.stack(data["action"])  # shape: (N, 6)

# action_is_pad をすべて False に初期化（N行の False ブール配列）
action_is_pad = np.zeros((action.shape[0],), dtype=bool)

# Dataset クラスに渡す
dataset = MyDataset(
    action=action,
    images=images,
    joint=joint,
    n_obs_steps=cfg.model.diffusion.n_obs_steps,
    horizon=cfg.model.diffusion.horizon,
)


dataloader = MyDataloader(dataset, cfg.data.split_ratio, cfg.data.batch_size, cfg.data.seed)


train_loader, val_loader, test_loader = dataloader.prepare_data()
# print(f"[DEBUG] train_dataset のサンプル数: {len(dataset)}")
# print(f"[DEBUG] train_loader のバッチ数: {len(train_loader)}")

# CNNで特徴量抽出
vision = VisionEncoder(
    channels=cfg.model.vision.channels,
    kernels=cfg.model.vision.kernels,
    strides=cfg.model.vision.strides,
    paddings=cfg.model.vision.paddings,
    latent_obs_dim=cfg.model.latent_obs_dim,
    mlp_hidden_dim=cfg.model.mlp.mlp_hidden_dim,
    n_mlp_layers=cfg.model.mlp.n_mlp_layers,
).to(device)

"""
    {    input_shapes=()
        "observation.image": [3, 240, 320],
        "observation.joint": [6],
        "action": [6]
    },
    output_shapes={
        "action": [6]
    },
"""


# diffusion pollicy
config = DiffusionConfig(
    n_obs_steps=cfg.model.diffusion.n_obs_steps,
    horizon=cfg.model.diffusion.horizon,
    n_action_steps=cfg.model.diffusion.n_action_steps,
    vision_backbone="resnet18",
    crop_shape=(240, 320),
    robot_state_feature_dim=cfg.model.policy.robot_state_feature_dim,
    vision_encoder_output_dim=cfg.model.policy.vision_encoder_output_dim,
    input_features={
        "observation.image": PolicyFeature(FeatureType.VISUAL, shape=(3, 240, 320)),
        "observation.state": PolicyFeature(FeatureType.STATE, shape=(6,)),
    },
    output_features={"action": PolicyFeature(FeatureType.ACTION, shape=(6,))},
)

diffusion_model = DiffusionModel(config).to(device)
policy = DiffusionPolicy(config, vision_encoder=vision, diffusion_model=diffusion_model).to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(
    policy.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay
)

trainer = TrainerSeparated(
    policy=policy,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    epoch=cfg.train.epoch,
    device=device,
    save_path=f"result/{cfg.wandb.train_name}/model",
)

trainer.train()  # トレーニングを開始

# print_model_structure(vision)
print_model_structure(policy)


best_epoch_path = cfg.logging.best_epoch_path

# 新しいフォルダを作成
best_epoch_folder = os.path.dirname(best_epoch_path)
os.makedirs(best_epoch_folder, exist_ok=True)

# 既存のエポックデータを読み込む
if os.path.exists(best_epoch_path):
    with open(best_epoch_path, "r") as f:
        best_epoch_data = yaml.safe_load(f)
    best_epoch = best_epoch_data["best_epoch"]

    # モデル読み込み
    # best_vision = load_file(f"result/{cfg.wandb.train_name}/model/vision_epoch_{best_epoch}.safetensors")
    best_policy = load_file(f"result/{cfg.wandb.train_name}/model/policy_epoch_{best_epoch}.safetensors")
    # vision.load_state_dict(best_vision)
    policy.load_state_dict(best_policy)
else:
    print(f"[warning] {best_epoch_path} not found. Skipping best model loading.")

# 必要に応じてモデルを保存する処理も追加
# best_epoch_data = {'best_epoch': current_epoch}
# with open(best_epoch_path, 'w') as f:
#     yaml.dump(best_epoch_data, f)

wandb.finish()
