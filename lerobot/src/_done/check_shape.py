import blosc2
import numpy as np


def load_blosc2(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        return blosc2.unpack_array2(f.read())


# 読み込み
joint_states = load_blosc2("data/merged/joint_states.blosc2")
action_states = load_blosc2("data/merged/action_states.blosc2")
image_states = load_blosc2("data/merged/image_states.blosc2")

# 形状を確認
print("joint_states.shape:", joint_states.shape)
print("action_states.shape:", action_states.shape)
print("image_states.shape:", image_states.shape)

print("action min/max:", action_states.min(), action_states.max())
print("joint min/max:", joint_states.min(), joint_states.max())
print("image min/max:", image_states.min(), image_states.max())
