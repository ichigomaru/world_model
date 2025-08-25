# import time
# import numpy as np
# import torch
# from lerobot_utils import Replay
# import os
# import cv2
# import imageio
# import yaml
# import matplotlib.pyplot as plt

# # 記録済み ground truth の読み込み
# # shape: (シーケンス数, ステップ数, 6)
# action_states = np.load("data/action_states_scaled.npy")
# sequence_index = 7  # 再生するシーケンス番号を指定
# actions = torch.tensor(action_states[sequence_index], dtype=torch.float32)

# # 正規化を戻す

# with open("conf/data_info.yaml", "r") as f:
#     stats = yaml.safe_load(f)

# min_vals = torch.tensor(stats["action_min"], dtype=torch.float32)
# max_vals = torch.tensor(stats["action_max"], dtype=torch.float32)

# def denormalize(x, min_val, max_val):
#     return ((x + 1) / 2) * (max_val - min_val) + min_val

# actions_denorm = denormalize(actions, min_vals, max_vals)

# # ロボット接続
# replay = Replay(
#     height=480,
#     width=640,
#     camera_id=(0,),
#     is_higher_port=False,
#     leader_port="/dev/tty.usbmodem57640257221",
#     follower_port="/dev/tty.usbmodem58370529971",
#     calibration_name="koch"
# )

# # アクションを再生
# # 記録処理を追加
# # アクション保存用

# save_dir = "output/ground_truth_replay"
# os.makedirs(save_dir, exist_ok=True)
# torch.save(actions_denorm, os.path.join(save_dir, "replayed_actions.pt"))
# print(f"アクションを {os.path.join(save_dir, 'replayed_actions.pt')} に保存しました。")

# # GIF用画像リスト
# images = []

# FPS = 5
# # joints vs replayed_actions の可視化用
# joints = []
# # アクションを再生（再実行用に別ループに移動して joint も記録）
# for action in actions_denorm:
#     obs = replay.get_observations()
#     img = obs["image.main"]  # torch.Tensor (3, H, W)
#     images.append(img.clone())

#     joint = torch.tensor(obs["observation.state"], dtype=torch.float32)
#     joints.append(joint.clone())

#     replay.send(action=action, fps=FPS)
#     time.sleep(1.0 / FPS)

# # GIF保存処理
# gif_path = os.path.join(save_dir, "replay.gif")
# images_np = []

# for step, img in enumerate(images):
#     img_np = img.permute(1, 2, 0).cpu().numpy()
#     img_np = img_np.copy()
#     img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
#     h, w, _ = img_bgr.shape

#     margin_top_bottom = 50
#     margin_sides = 30
#     canvas = np.ones((h + margin_top_bottom * 2, w + margin_sides * 2, 3), dtype=np.uint8) * 255
#     canvas[margin_top_bottom:margin_top_bottom + h, margin_sides:margin_sides + w, :] = img_bgr

#     text = f"Timestep {step}"
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 1.0
#     thickness = 2
#     text_color = (0, 0, 0)
#     text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
#     text_width, text_height = text_size
#     text_x = (w + margin_sides * 2 - text_width) // 2
#     text_y = (margin_top_bottom + text_height) // 2

#     cv2.putText(canvas, text, (text_x, text_y), font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)
#     img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
#     images_np.append(img_rgb)

# imageio.mimsave(gif_path, images_np, fps=FPS)
# print(f"GIFを保存しました: {gif_path}")

# # joints保存
# joints_tensor = torch.stack(joints)  # (NUM_STEPS, 6)
# torch.save(joints_tensor, os.path.join(save_dir, "joints.pt"))

# # グラフ保存
# joints_np = joints_tensor.numpy()
# actions_np = actions_denorm.numpy()


# fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
# for i in range(6):
#     axes[i].plot(actions_np[:, i], label='Action (GT)', linestyle='--')
#     axes[i].plot(joints_np[:, i], label='Joint', linestyle='-')
#     axes[i].set_ylabel(f'Joint {i+1}')
#     if i == 0:
#         axes[i].legend(loc='upper right')
# axes[-1].set_xlabel("Time step")
# plt.suptitle("Joint vs Ground Truth Action")
# plt.tight_layout()
# prediction_plot_path = os.path.join(save_dir, 'joint_vs_gt_action.png')
# plt.savefig(prediction_plot_path)
# plt.close()
# print(f"Saved joint vs ground truth action: {prediction_plot_path}")
