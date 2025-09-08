import os
import random
import blosc2

import numpy as np
import torch
import torch.nn as nn
import wandb
import yaml
from box import Box
from datasets import load_from_disk
from safetensors.torch import load_file

from src.model.world_model.encoder import VisionEncoder
from src.model.world_model.RSSM import RSSM
from src.model.world_model.decoder import VisionDecoder
from src.model.world_model.WorldModel import WorldModel
from src.trainer.trainer_worldmodel import Trainer

from src.dataloader.dataloader import MyDataloader
from src.dataset.dataset import MyDataset
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
)  
os.makedirs("output", exist_ok=True)

with open("conf/conf.yaml", "r") as yml:
    cfg = Box(yaml.safe_load(yml))

os.makedirs(f"result/{cfg.wandb.train_name}", exist_ok=True)
wandb.init(project=cfg.wandb.project_name, config=cfg.wandb.config, name=cfg.wandb.train_name)


data_dir = "data_merged1"

images = blosc2.load_array(os.path.join(data_dir, "images.blosc2"))
joint = blosc2.load_array(os.path.join(data_dir, "states.blosc2"))
action = blosc2.load_array(os.path.join(data_dir, "actions.blosc2"))

action_is_pad = np.zeros((action.shape[0],), dtype=bool)

dataset = MyDataset(
    action=action,
    images=images,
    joint=joint,
    sequence_length=cfg.data.sequence_length
)

dataloader = MyDataloader(dataset, cfg.data.split_ratio, cfg.data.batch_size, cfg.data.seed)


train_loader, val_loader, test_loader = dataloader.prepare_data()

encoder = VisionEncoder(
    channels=cfg.model.vision.channels,
    kernels=cfg.model.vision.kernels,
    strides=cfg.model.vision.strides,
    paddings=cfg.model.vision.paddings,
    latent_obs_dim=cfg.model.latent_obs_dim,
    mlp_hidden_dim=cfg.model.mlp.mlp_hidden_dim,
    n_mlp_layers=cfg.model.mlp.n_mlp_layers,
).to(device)

rssm = RSSM(action_size=6, config=cfg).to(device) 

decoder = VisionDecoder(
    channels=cfg.model.vision.channels,
    kernels=cfg.model.vision.kernels,
    strides=cfg.model.vision.strides,
    paddings=cfg.model.vision.paddings,
    latent_obs_dim=cfg.parameters.dreamer.deterministic_size + cfg.parameters.dreamer.stochastic_size, # RSSMの確率的状態の次元
    mlp_hidden_dim=cfg.model.mlp.mlp_hidden_dim,
    n_mlp_layers=cfg.model.mlp.n_mlp_layers,
).to(device)

world_model = WorldModel(encoder, rssm, decoder).to(device)
optimizer = torch.optim.Adam(world_model.parameters(), lr=cfg.train.learning_rate)

trainer = Trainer(
    model=world_model,  
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    epoch=cfg.train.epoch,
    device=device,
    save_path=f"result/{cfg.wandb.train_name}/model",
)

trainer.train()  # トレーニングを開始

print_model_structure(world_model)

best_epoch_path = os.path.join(f"result/{cfg.wandb.train_name}/model", 'best_epoch.yaml')

best_epoch_folder = os.path.dirname(best_epoch_path)
os.makedirs(best_epoch_folder, exist_ok=True)

if os.path.exists(best_epoch_path):
    with open(best_epoch_path, "r") as f:
        best_epoch_data = yaml.safe_load(f)
    best_epoch = best_epoch_data["best_epoch"]

    best_model_path = os.path.join(f"result/{cfg.wandb.train_name}/model", "world_model_best.safetensors")

    best_model_weights = load_file(best_model_path)
    world_model.load_state_dict(best_model_weights)

else:
    print(f"[warning] {best_epoch_path} not found. Skipping best model loading.")

wandb.finish()
