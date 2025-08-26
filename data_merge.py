import os
import torch
import numpy as np
import blosc2
from datasets import Dataset, Features, Value, Array3D, Sequence
from huggingface_hub import HfApi

# blosc2読み込み関数（例）
def load_blosc2(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        return blosc2.unpack_array2(f.read())
    
joint_right = load_blosc2("data/right/joint_states.blosc2")
action_right = load_blosc2("data/right/action_states.blosc2")
image_right = load_blosc2("data/right/images_states.blosc2")

joint_left = load_blosc2("data/left/joint_states.blosc2")
action_left = load_blosc2("data/left/action_states.blosc2")
image_left = load_blosc2("data/left/images_states.blosc2")

# Ensure all datasets have same length
assert joint_right.shape[1:] == joint_left.shape[1:]
assert action_right.shape[1:] == action_left.shape[1:]
assert image_right.shape[2:] == image_left.shape[2:]

# Determine the number of episodes
num_right = joint_right.shape[0]
num_left = joint_left.shape[0]
min_len = min(num_right, num_left)

# Interleave right and left data
def interleave(a, b):
    interleaved = np.empty((min_len * 2, *a.shape[1:]), dtype=a.dtype)
    interleaved[0::2] = a[:min_len]
    interleaved[1::2] = b[:min_len]
    return interleaved

joint_merged = interleave(joint_right, joint_left)
action_merged = interleave(action_right, action_left)
image_merged = interleave(image_right, image_left)

# Save merged data
os.makedirs("data/merged", exist_ok=True)
np.save("data/merged/joint_states.npy", joint_merged)
np.save("data/merged/action_states.npy", action_merged)
np.save("data/merged/images_states.npy", image_merged)

print(joint_merged.shape)
print(joint_merged.dtype)
print(action_merged.shape)
print(image_merged.shape)
