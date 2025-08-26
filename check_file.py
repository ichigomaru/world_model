import numpy as np
import blosc2
from datasets import Dataset, Features, Value, Array3D, Sequence
from huggingface_hub import HfApi

with open("data/merged/joint_states.blosc2", "rb") as f:
    arr = blosc2.unpack_array2(f.read())
print(arr.shape)  # ← (82, 50, 6) になるのが理想