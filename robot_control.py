import torch
import yaml
from box import Box
from safetensors.torch import load_file
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torchvision.transforms as T

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "lerobot")
sys.path.insert(0, project_root)

from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy, DiffusionModel # type: ignore
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig # type: ignore
from lerobot.configs.types import FeatureType, PolicyFeature # type: ignore
import time
from safetensors.torch import load_file
import blosc2
import cv2
import imageio

from src.model.vision import VisionEncoder
from datasets import load_from_disk
from collections import deque
from torchvision import transforms
import lerobot_utils

with open('./lerobot/conf/conf.yaml', 'r') as yml:
    cfg = Box(yaml.safe_load(yml))

device = torch.device(cfg.wandb.config.device if torch.cuda.is_available() else 'cpu')

episode_index = 8  
target_episode = 60

# ロボット初期化
replay = lerobot_utils.Replay(
    height=480,
    width=640,
    camera_id=(0,),
    is_higher_port=False,
    leader_port="/dev/tty.usbmodem57640257221",
    follower_port="/dev/tty.usbmodem58370529971",
    calibration_name="koch"
)

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

#diffusion pollicy
config = DiffusionConfig(
    n_obs_steps=cfg.model.diffusion.n_obs_steps,
    horizon=cfg.model.diffusion.horizon,
    n_action_steps=cfg.model.diffusion.n_action_steps,
    vision_backbone="resnet18",
    crop_shape=(240, 320),
    input_features={
        "observation.image": PolicyFeature(FeatureType.VISUAL, shape=(3, 240, 320)),
        "observation.state": PolicyFeature(FeatureType.STATE, shape=(6,))
    },
    output_features={
        "action": PolicyFeature(FeatureType.ACTION, shape=(6,))
    }
    )

diffusion_model = DiffusionModel(config).to(device)
policy = DiffusionPolicy(config, vision_encoder=vision, diffusion_model=diffusion_model).to(device)

#初期姿勢を固定
def load_blosc2(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        return blosc2.unpack_array2(f.read())
    
joint_states = load_blosc2("data/left/joint_states.blosc2")

dataset = load_from_disk("data_merged")
episode_data = dataset.filter(
    lambda x: x["episode_index"] == target_episode
)
episode_data_sorted = sorted(episode_data, key=lambda x: x["frame_index"])
first_frame = episode_data_sorted[0]

initial_joint = torch.tensor(first_frame["observation.state"], dtype=torch.float32)

replay.send(action=initial_joint, fps=5)
print("Sent initial joint position to robot.")

time.sleep(2)  # 2秒待機してから推論開始

# ベストモデルの読み込み
best_epoch_path = f"lerobot/result/{cfg.wandb.train_name}/model/best_epoch.yaml"
if os.path.exists(best_epoch_path):
    with open(best_epoch_path, 'r') as f:
        best_epoch_data = yaml.safe_load(f)
    best_epoch = best_epoch_data['best_epoch']

    policy_path = f'lerobot/result/{cfg.wandb.train_name}/model/policy_epoch_{best_epoch}.safetensors'

    policy.load_state_dict(load_file(policy_path))
else:
    raise FileNotFoundError(f"{best_epoch_path} が存在しない。トレーニングでbest_epoch.yamlを保存したか確認")


# 正規化準備
with open('conf/data_info.yaml', 'r') as f:
    norm_stats = yaml.safe_load(f)

# joint の max/min 取得
min_vals_joint = torch.tensor(norm_stats['follower_min'], dtype=torch.float32).to(device)
max_vals_joint = torch.tensor(norm_stats['follower_max'], dtype=torch.float32).to(device)

# action の max/min 取得
min_vals_action = torch.tensor(norm_stats['action_min'], dtype=torch.float32).to(device)
max_vals_action = torch.tensor(norm_stats['action_max'], dtype=torch.float32).to(device)

# 正規化関数スケーリング
def normalize(x, min_val, max_val):
    x = torch.clamp(x, min=min_val, max=max_val)    # 外れ値をカット
    return (x - min_val) / (max_val - min_val) * 2 - 1

# 逆正規化
def denormalize(x, min_val, max_val):
    return ((x + 1) / 2) * (max_val - min_val) + min_val



vision.eval()
policy.eval()

images = []
joints = []
image_features = []

total_steps = cfg.run.steps
FPS = cfg.run.fps
n_obs_steps = cfg.model.diffusion.n_obs_steps
horizon = cfg.model.diffusion.horizon
n_action_steps = cfg.model.diffusion.n_action_steps

obs_buffer = deque(maxlen=n_obs_steps)  # 観測データのバッファ

for i in range(n_obs_steps):
    frame_data = episode_data_sorted[i]
    initial_obs = {
        "image.main": frame_data["observation.image"],
        "observation.state": frame_data["observation.state"]
    }
    obs_buffer.append(initial_obs)


#メインループ
num_loops = (total_steps) // n_action_steps
pred_actions_log = []
obs_joint_log = []
all_images = []

with torch.no_grad():
    for i in range(num_loops):

        images = []
        states = []
        transform = transforms.Resize((240, 320))

        for obs in obs_buffer:

            img = torch.tensor(obs["image.main"]).float() / 255.0
            img = transform(img)
            images.append(img)

            state = torch.tensor(obs["observation.state"], dtype=torch.float32)
            state = normalize(state, min_vals_joint, max_vals_joint)
            states.append(state)

        batch = {
            "observation.images": torch.stack(images).unsqueeze(0).to(device),  # (n_obs_steps, 3, 240, 320)
            "observation.state": torch.stack(states).unsqueeze(0).to(device)    # (n_obs_steps, 6)
        }

        # == 推論 ==
        B, T, C, H, W = batch["observation.images"].shape
        images_flat = batch["observation.images"].reshape(B * T, C, H, W)
        features_flat = vision(images_flat)  # (B*T, feature_dim)
        image_features = features_flat.reshape(B, T, -1)  # (B, T, feature_dim)
        batch["image_features"] = image_features

        pred_actions = policy.diffusion.generate_actions(batch)[0] #horizon分の行動を予測

        # == 行動の実行 ==
        for i in range(n_action_steps):
            action = pred_actions[i].cpu()
            action_denorm = denormalize(action, min_vals_action, max_vals_action)  # 逆正規化

            replay.send(action=action_denorm, fps=FPS)
            new_obs = replay.get_observations() #新しい観測を獲得
            obs_buffer.append(new_obs) #バッファを更新(古いのが消える)

            all_images.append(torch.tensor(new_obs["image.main"]).float() / 255.0)
            pred_actions_log.append(action_denorm.cpu())
            obs_joint_log.append(torch.tensor(new_obs["observation.state"], dtype=torch.float32)) 




# データ保存用コード
# 保存用ディレクトリ
train_name_safe = cfg.wandb.train_name.replace(":", "_")
save_root = f'output/{train_name_safe}'
raw_dir = os.path.join(save_root, 'raw')

os.makedirs(raw_dir, exist_ok=True)


# images の GIF保存
gif_save_path = os.path.join(save_root, f'images[{episode_index}].gif')

margin_top_bottom = 50     # 上下余白
margin_sides = 40  # 左右余白

images_with_text = []

for step, img in enumerate(all_images):
    img_np = img.permute(1, 2, 0).cpu().numpy()  #(H, W, 3)
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    #img_np = img_np.copy()

    # OpenCV用にBGR変換
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    h, w, _ = img_bgr.shape

    # 上下左右に余白
    canvas = np.ones((h + margin_top_bottom * 2, w + margin_sides * 2, 3), dtype=np.uint8) * 255

    # 元画像を中央に貼り付け
    canvas[margin_top_bottom : margin_top_bottom + h, margin_sides : margin_sides + w, :] = img_bgr

    # テキストを中央に描画
    text = f"Timestep {step}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    text_color = (0, 0, 0)

    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_width, text_height = text_size
    text_x = (w + margin_sides * 2 - text_width) // 2
    text_y = (margin_top_bottom + text_height) // 2

    cv2.putText(
        canvas, text,
        (text_x, text_y),
        font, font_scale,
        text_color, thickness,
        lineType=cv2.LINE_AA
    )

    # BGR → RGBに戻す
    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    images_with_text.append(img_rgb)

imageio.mimsave(gif_save_path, images_with_text, fps=FPS)
print(f"Saved GIF: {gif_save_path}")




# == 予測と実際の関節角度の可視化 ==
pred_actions_np = torch.stack(pred_actions_log).numpy()  # (NUM_STEPS, 6)
obs_joint_np = torch.stack(obs_joint_log).numpy()  # (NUM_STEPS, 6)

fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
for i in range(6):
    axes[i].plot(pred_actions_np[:, i], label='prediction', linestyle='--')
    axes[i].plot(obs_joint_np[:, i], label='obs_joint', linestyle='-')
    axes[i].set_ylabel(f'joint{i+1}')
    if i == 0:
        axes[i].legend(loc='upper right')

axes[-1].set_xlabel("Time step") #横軸のメモリ
axes[-1].xaxis.set_major_locator(ticker.MultipleLocator(10)) # 主目盛りを10刻み
axes[-1].xaxis.set_minor_locator(ticker.MultipleLocator(1)) # 副目盛りを1刻み

# 主目盛りのフォントサイズや長さ調整
axes[-1].tick_params(axis='x', which='major', length=10, labelsize=10)
axes[-1].tick_params(axis='x', which='minor', length=5)

plt.suptitle(f"prediction_&_obs_joint_data[{episode_index}]")
plt.tight_layout()
prediction_plot_path = os.path.join(save_root, f'prediction_&_obs_joint_data[{episode_index}].png')
plt.savefig(prediction_plot_path)
plt.close()
print(f"Saved action_prediction_&_obs_joint_data[{episode_index}]: {prediction_plot_path}")

# # image_features の可視化
# image_features_np = image_features_tensor.numpy()   #(NUM_STEPS, latent_dim)
# plt.figure(figsize=(12, 6))
# for i in range(image_features_np.shape[1]):
#     plt.plot(image_features_np[:, i], label=f'feature_{i}')
# plt.title('Image Features Over Time')
# plt.xlabel('Time Step')
# plt.ylabel('Feature Value')
# plt.legend(fontsize=6, ncol=4)
# plt.tight_layout()
# features_plot_path = os.path.join(save_root, 'image_features.png')
# plt.savefig(features_plot_path)
# plt.close()
# print(f"Saved Image Features: {features_plot_path}")

# actionp_prediction と ground_truth の可視化