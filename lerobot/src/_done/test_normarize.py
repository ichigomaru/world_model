import torch
import yaml
from box import Box

with open("./conf/conf.yaml", "r") as yml:
    cfg = Box(yaml.safe_load(yml))
device = torch.device(cfg.wandb.config.device if torch.cuda.is_available() else "cpu")


# 正規化準備
with open("conf/data_info.yaml", "r") as f:
    norm_stats = yaml.safe_load(f)
# joint の max/min 取得
min_vals_joint = torch.tensor(norm_stats["follower_min"], dtype=torch.float32).to(device)
max_vals_joint = torch.tensor(norm_stats["follower_max"], dtype=torch.float32).to(device)
# action の max/min 取得
min_vals_action = torch.tensor(norm_stats["action_min"], dtype=torch.float32).to(device)
max_vals_action = torch.tensor(norm_stats["action_max"], dtype=torch.float32).to(device)

print("min_vals_joint:", min_vals_joint)
print("max_vals_joint:", max_vals_joint)
print("min_vals_action:", min_vals_action)
print("max_vals_action:", max_vals_action)
