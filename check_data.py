import numpy as np
from datasets import load_from_disk
from safetensors.torch import load_file

data = load_from_disk("data_merged")

images = np.stack(data["observation.image"])  # shape: (N, 3, 240, 320)
joint = np.stack(data["observation.state"])  # shape: (N, 6)
action = np.stack(data["action"])  # shape: (N, 6)

print(images.shape)
print(joint.shape)
print(action.shape)

# 48×64に修正