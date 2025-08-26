'''
import hydra
from src.conf import TrainConfig

@hydra.main(config_path="conf", confg_name="train", version_base="1.1")
def main(cfg: TrainConfig):


if __name__ == "__main__":
    main()
'''
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import yaml
import wandb
import random
from box import Box
from safetensors.torch import load_file
from tqdm import tqdm

from model.vision_RNN import VisionEncoder
from src.model.policy import RNN
from src.trainer.trainer import TrainerSeparated
from src.dataset.dataset import MyDataset
from src.dataloader.dataloader import MyDataloader
from src.utils.print_model import print_model_structure
from src.utils.read_blosc2 import load_blosc2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#シードの固定
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)

os.chdir(os.path.dirname(os.path.abspath(__file__)))  #ch = change このスクリプトと同じディレクトリをカレントディレクトリにする __file__はこのスクリプトのパスを取得する つまりkensyu_3に移動
os.makedirs('output', exist_ok=True)

with open('conf/conf.yaml', 'r') as yml:
    cfg = Box(yaml.safe_load(yml))

os.makedirs(f'result/{cfg.wandb.train_name}', exist_ok=True)
wandb.init(project=cfg.wandb.project_name, config=cfg.wandb.config, name=cfg.wandb.train_name)


action = load_blosc2('data/merged/action_states.blosc2')
images = load_blosc2('data/merged/image_states.blosc2')
joint = load_blosc2('data/merged/joint_states.blosc2')

# print("action shape:", action.shape)
# print(action)

dataset = MyDataset(action, images, joint)

# print("データ数:", len(dataset))         # __len__ が呼ばれる
# print("最初のデータ:", dataset[0])   # __getitem__ が呼ばれる

dataloader = MyDataloader(dataset,
                        cfg.data.split_ratio,
                        cfg.data.batch_size,
                        cfg.data.seed
                        )


train_loader, val_loader, test_loader = dataloader.prepare_data()

# for action, images, joint in train_loader:
#     print("images.shape:", images.shape)
#     break

#CNNで特徴量抽出
vision = VisionEncoder(
    channels=cfg.model.vision.channels,
    kernels=cfg.model.vision.kernels,
    strides=cfg.model.vision.strides,
    paddings=cfg.model.vision.paddings,
    latent_obs_dim=cfg.model.latent_obs_dim,
    mlp_hidden_dim=cfg.model.mlp.mlp_hidden_dim,
    n_mlp_layers=cfg.model.mlp.n_mlp_layers
)

policy_input_dim = cfg.model.latent_obs_dim + 6
policy = RNN(
    input_dim = policy_input_dim,
    hidden_dim = cfg.model.policy.hidden_dim,
    output_dim = 6
).to(device)

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(
    list(vision.parameters()) + list(policy.parameters()),
    lr=cfg.train.learning_rate,
    weight_decay=cfg.train.weight_decay
)

#print("データセット長さ:", len(dataset))

trainer = TrainerSeparated(
    vision=vision,
    policy=policy,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epoch=cfg.train.epoch,
    device=device,
    save_path=f"result/{cfg.wandb.train_name}/model"
)

print_model_structure(vision)
print_model_structure(policy)

#print("trainerの前のimage.shape", images.shape)

# --- Vision モデルの重み変化をチェックする ---

# 学習前の重みを保存（例として最初のConv層の重み）
before_weights = vision.encoder[0].weight.clone().detach()
trainer.train()
# 学習後の重みを再取得
after_weights = vision.encoder[0].weight.clone().detach()

# 差分のノルムを計算
diff = torch.norm(after_weights - before_weights).item()
print(f"学習前後の重み変化（L2ノルム）: {diff:.6f}")

# 変化が小さすぎる場合、学習されていない可能性あり
if diff < 1e-5:
    print("⚠️ Vision モデルの重みがほとんど更新されていません。視覚情報が学習に使われていない可能性があります。")
else:
    print("✅ Vision モデルの重みは更新されています。視覚情報が学習に寄与しています。")

best_epoch_path = cfg.logging.best_epoch_path

# 新しいフォルダを作成
best_epoch_folder = os.path.dirname(best_epoch_path)
os.makedirs(best_epoch_folder, exist_ok=True)

# 既存のエポックデータを読み込む
if os.path.exists(best_epoch_path):
    with open(best_epoch_path, 'r') as f:
        best_epoch_data = yaml.safe_load(f)
    best_epoch = best_epoch_data['best_epoch']
    
    # モデル読み込み
    best_vision = load_file(f"result/{cfg.wandb.train_name}/model/vision_epoch_{best_epoch}.safetensors")
    best_policy = load_file(f"result/{cfg.wandb.train_name}/model/policy_epoch_{best_epoch}.safetensors")
    vision.load_state_dict(best_vision)
    policy.load_state_dict(best_policy)
else:
    print(f"[warning] {best_epoch_path} not found. Skipping best model loading.")

# 必要に応じてモデルを保存する処理も追加
# best_epoch_data = {'best_epoch': current_epoch}
# with open(best_epoch_path, 'w') as f:
#     yaml.dump(best_epoch_data, f)

wandb.finish()

'''
# --- 潜在ベクトルの min/max をプロット ---
vision.eval()
latent_list = []

with torch.no_grad():
    for i, (images, joints, actions) in enumerate(train_loader):
        images = images.to(device).float()

        if images.dim() == 5:
            B, T, C, H, W = images.shape
            images = images.view(B * T, C, H, W)
        
        print("images.shape:", images.shape)
        features = vision(images)  # (B, latent_obs_dim)
        print("features.shape:", features.shape)
        latent_list.append(features.cpu())
        #print(latent_list)
        #print(images)

        if i >= 10:  # 最初の10バッチのみ使う（多すぎると重い）
            break

latent_all = torch.cat(latent_list, dim=0)  # (N, latent_obs_dim)
latent_min = latent_all.min(dim=0).values
latent_max = latent_all.max(dim=0).values

# プロット
plt.figure(figsize=(10, 4))
plt.plot(latent_min.numpy(), label='min')
plt.plot(latent_max.numpy(), label='max')
plt.title('Latent Feature Value Ranges (min/max per dimension)')
plt.xlabel('Latent Dimension')
plt.ylabel('Value')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('output/latent_range.png')
#plt.close()
#print("Saved latent feature range plot to output/latent_range.png")


# --- テストデータの latent を PCA で 2次元プロット ---
from sklearn.decomposition import PCA
from torchvision.transforms import Resize

print("Computing PCA of test latent features...")
vision.eval()
test_latents = []
with torch.no_grad():
    for i, (images, joints, actions) in enumerate(test_loader):
        images = images.to(device).float()
        if images.dim() == 5:
            B, T, C, H, W = images.shape
            images = images.view(B * T, C, H, W)
        features = vision(images)
        test_latents.append(features.cpu())
        if i >= 10:
            break

test_latent_all = torch.cat(test_latents, dim=0)  # (N_test, latent_dim)

# --- 実ロボット画像データからのlatentも可視化 ---
train_name_safe = cfg.wandb.train_name
real_image_path = 'output/{}/robot_test/raw/images.pt'.format(train_name_safe)
if os.path.exists(real_image_path):
    real_images = torch.load(real_image_path).float().to(device) / 255.0

    if real_images.dim() == 5:
        B, T, C, H, W = real_images.shape
        real_images = real_images.view(B * T, C, H, W)

    resize = Resize((48, 64))
    real_images = resize(real_images)

    with torch.no_grad():
        real_latents = vision(real_images).cpu()
else:
    print(f"[warning] Real robot image file not found: {real_image_path}")
    real_latents = None

# 既に train 用 latent_all があるので、それと結合して PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
if real_latents is not None:
    latent_combined = torch.cat([latent_all, test_latent_all, real_latents], dim=0).numpy()
else:
    latent_combined = torch.cat([latent_all, test_latent_all], dim=0).numpy()
latent_all_pca = pca.fit_transform(latent_all.numpy())
latent_test_pca = pca.transform(test_latent_all.numpy())
if real_latents is not None:
    latent_real_pca = pca.transform(real_latents.numpy())

latent_2d = np.concatenate([latent_all_pca, latent_test_pca], axis=0)
if real_latents is not None:
    latent_2d = np.concatenate([latent_2d, latent_real_pca], axis=0)
    
n_train = latent_all.shape[0]
train_2d = latent_2d[:n_train]
test_2d = latent_2d[n_train:]
if real_latents is not None:
    test_len = test_latent_all.shape[0]
    real_2d = latent_2d[n_train + test_len:]

# プロット
plt.figure(figsize=(8, 6))
plt.scatter(train_2d[:, 0], train_2d[:, 1], label='Train', alpha=0.5, s=10)
plt.plot(test_2d[:, 0], test_2d[:, 1], label='Test (seq)', alpha=0.8, color='red', linewidth=2)
if real_latents is not None:
    plt.plot(real_2d[:, 0], real_2d[:, 1], label='Real Robot', alpha=0.9, color='green', linewidth=2)
plt.title('PCA of Latent Features')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('output/pca_latent_features.png')
plt.close()
print("Saved PCA latent feature plot to output/pca_latent_features.png")
'''