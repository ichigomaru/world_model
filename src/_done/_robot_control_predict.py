import torch
import yaml
from box import Box
from safetensors.torch import load_file
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2
import torchvision.transforms as T

from model.vision_RNN import VisionEncoder
from src.model.policy import RNN
import lerobot_utils

#モデルを動かして一度作ったデータを呼び出して再生

with open('./conf/conf.yaml', 'r') as yml:
    cfg = Box(yaml.safe_load(yml))

device = torch.device(cfg.wandb.config.device if torch.cuda.is_available() else 'cpu')

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

policy = RNN(
    input_dim=cfg.model.latent_obs_dim + 6,
    hidden_dim=cfg.model.policy.hidden_dim,
    output_dim=6
).to(device)

# ベストモデルの読み込み
best_epoch_path = f"result/{cfg.wandb.train_name}/model/best_epoch.yaml"
if os.path.exists(best_epoch_path):
    with open(best_epoch_path, 'r') as f:
        best_epoch_data = yaml.safe_load(f)
    best_epoch = best_epoch_data['best_epoch']

    vision_path = f'result/{cfg.wandb.train_name}/model/vision_epoch_{best_epoch}.safetensors'
    policy_path = f'result/{cfg.wandb.train_name}/model/policy_epoch_{best_epoch}.safetensors'

    vision.load_state_dict(load_file(vision_path))
    policy.load_state_dict(load_file(policy_path))
else:
    raise FileNotFoundError(f"{best_epoch_path} が存在しません。トレーニングでbest_epoch.yamlを保存したか確認してください。")

#推論モードに
vision.eval()
policy.eval()

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

# 正規化準備
with open('conf/data_info.yaml', 'r') as f:
    norm_stats = yaml.safe_load(f)
# joint の max/min 取得
min_vals_joint = torch.tensor(norm_stats['follower_min'], dtype=torch.float32).to(device)
max_vals_joint = torch.tensor(norm_stats['follower_max'], dtype=torch.float32).to(device)
# action の max/min 取得
min_vals_action = torch.tensor(norm_stats['action_min'], dtype=torch.float32).to(device)
max_vals_action = torch.tensor(norm_stats['action_max'], dtype=torch.float32).to(device)
# 正規化関数 [-1, 1]にスケーリング
def normalize(x, min_val, max_val):
    x = torch.clamp(x, min=min_val, max=max_val)    # 外れ値をカット
    return (x - min_val) / (max_val - min_val + 1e-8) * 2 - 1
# 逆正規化
def denormalize(x, min_val, max_val):
    return ((x + 1) / 2) * (max_val - min_val) + min_val


# 推論ループ
NUM_STEPS = cfg.run.steps
FPS = cfg.run.fps

images = []
joints = []
image_features = []
action_predictions = []

train_name_safe = cfg.wandb.train_name.replace(":", "_")
original_actions = torch.load('output/{}/robot_test/raw/action_predictions.pt'.format(train_name_safe))

print(f"Start robot control for {NUM_STEPS} steps at {FPS} FPS.")
policy.initialize(batch_size=1, device=device)

with torch.no_grad():
    for step in range(NUM_STEPS):

        # 観測取得
        obs = replay.get_observations()

        # image
        image_original = obs["image.main"]  # (3, H, W), unit8
        images.append(image_original.clone()) # image 保存
        transform = T.Compose([
            T.Resize((48, 64)),
        ])
        image = image_original.float() / 255.0  # (3, 480, 640), float32, [0, 1]に正規化
        image = transform(image)      # (3, 48, 64)にリサイズ
        image_input = image.unsqueeze(0).to(device)   # (1, 3, 48, 64)

        # joint
        joint = torch.tensor(obs["observation.state"], dtype=torch.float32)  # (6, )
        joints.append(joint.clone())  # joint 保存
        joint_input = joint.unsqueeze(0).to(device)   # (1, 6)

        # モデル推論はスキップ
        # image_feature = vision(image_input)  # (1, latent_dim)
        # image_features.append(image_feature.squeeze(0).clone().cpu())   # (latent_dim, )

        # joint_norm = normalize(joint_input, min_vals_joint, max_vals_joint)

        # policy_input = torch.cat([image_feature, joint_norm], dim=-1).unsqueeze(1)  # (1, 1, latent_dim+6)
        # action_prediction = policy(policy_input)  # (1, 1, 6)
        # action_prediction = action_prediction.squeeze(0).squeeze(0).cpu()  # (6,)

        # action_prediction_denorm = denormalize(action_prediction, min_vals_action, max_vals_action)
        # action_predictions.append(action_prediction_denorm.clone())    # action_prediction 保存

        action_to_send = original_actions[step].to(device)

        # ロボットに送信
        replay.send(action=action_to_send, fps=FPS)


# データ保存用コード
# 保存用ディレクトリ
save_root = f'output/{train_name_safe}/robot_test'
raw_dir = os.path.join(save_root, 'raw')

os.makedirs(raw_dir, exist_ok=True)


# rawデータ保存
raw_dir = os.path.join(save_root, 'raw')
os.makedirs(save_root, exist_ok=True)

images_tensor = torch.stack(images)     # (NUM_STEPS, 3, H, W)
torch.save(images_tensor, os.path.join(raw_dir, 'images.pt'))

joints_tensor = torch.stack(joints)  # (NUM_STEPS, 6)
torch.save(joints_tensor, os.path.join(raw_dir, 'joints.pt'))

image_features_tensor = torch.stack(image_features)  # (NUM_STEPS, latent_dim)
torch.save(image_features_tensor, os.path.join(raw_dir, 'image_features.pt'))

action_predictions_tensor = torch.stack(action_predictions)  # (NUM_STEPS, 6)
torch.save(action_predictions_tensor, os.path.join(raw_dir, 'action_predictions.pt'))

print(f"Saved raw data in {raw_dir}")


# images の GIF保存
gif_save_path = os.path.join(save_root, 'images.gif')

margin_top_bottom = 50     # 上下余白
margin_sides = 30  # 左右余白

images_with_text = []

for step, img in enumerate(images):
    img_np = img.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    img_np = img_np.copy()

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


# image_features の可視化
image_features_np = image_features_tensor.numpy()   # (NUM_STEPS, latent_dim)
plt.figure(figsize=(12, 6))
for i in range(image_features_np.shape[1]):
    plt.plot(image_features_np[:, i], label=f'feature_{i}')
plt.title('Image Features Over Time')
plt.xlabel('Time Step')
plt.ylabel('Feature Value')
plt.legend(fontsize=6, ncol=4)
plt.tight_layout()
features_plot_path = os.path.join(save_root, 'image_features.png')
plt.savefig(features_plot_path)
plt.close()
print(f"Saved Image Features: {features_plot_path}")

# joints vs action_predictions の可視化
joints_np = joints_tensor.numpy()  # (NUM_STEPS, 6)
action_predictions_np = action_predictions_tensor.numpy()  # (NUM_STEPS, 6)

fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
for i in range(6):
    axes[i].plot(action_predictions_np[:, i], label='action Prediction', linestyle='--')
    axes[i].plot(joints_np[:, i], label='joint', linestyle='-')
    axes[i].set_ylabel(f'joint{i+1}')
    if i == 0:
        axes[i].legend(loc='upper right')
axes[-1].set_xlabel("Time step")
plt.suptitle("joint vs action Prediction")
plt.tight_layout()
prediction_plot_path = os.path.join(save_root, 'joint_vs_action.png')
plt.savefig(prediction_plot_path)
plt.close()
print(f"Saved joint vs action: {prediction_plot_path}")