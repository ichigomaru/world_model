import os
import numpy as np
import blosc2

# ---------- Blosc2 の保存／読込ユーティリティ ----------
def save_blosc2(arr: np.ndarray, path: str) -> None:
    """
    NumPy 配列を Blosc2 形式で保存する
    """
    with open(path, "wb") as f:
        f.write(blosc2.pack_array2(arr))

def load_blosc2(path: str) -> np.ndarray:
    """
    Blosc2 形式ファイルを読み込んで NumPy 配列に戻す
    """
    with open(path, "rb") as f:
        return blosc2.unpack_array2(f.read())

# ---------- left / right の 40 シーケンスずつを結合して書き出し ----------
save_dir = "data/merged"                 # 例：保存先ディレクトリ
os.makedirs(save_dir, exist_ok=True)

# joint ----------------------------------------------------------------------
right_joint = load_blosc2("data/right/joint_states.blosc2")[:40]   # (40, 50, 6)
left_joint  = load_blosc2("data/left/joint_states.blosc2")[:40]    # (40, 50, 6)
merged_joint = np.concatenate([right_joint, left_joint], axis=0)   # (80, 50, 6)
save_blosc2(merged_joint, os.path.join(save_dir, "joint_states.blosc2"))

# action ---------------------------------------------------------------------
right_action = load_blosc2("data/right/action_states.blosc2")[:40]
left_action  = load_blosc2("data/left/action_states.blosc2")[:40]
merged_action = np.concatenate([right_action, left_action], axis=0)
save_blosc2(merged_action, os.path.join(save_dir, "action_states.blosc2"))

# image ----------------------------------------------------------------------
right_image = load_blosc2("data/right/image_states.blosc2")[:40]   # (40, 50, 3, 48, 64)
left_image  = load_blosc2("data/left/image_states.blosc2")[:40]
merged_image = np.concatenate([right_image, left_image], axis=0)   # (80, 50, 3, 48, 64)
save_blosc2(merged_image, os.path.join(save_dir, "image_states.blosc2"))

print("✅ Blosc2 形式で保存が完了しました。")