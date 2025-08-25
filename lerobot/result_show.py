import matplotlib.pyplot as plt
import torch
import yaml
from box import Box
from datasets import load_from_disk
from safetensors.torch import load_file

from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.configs.types import FeatureType
from src.dataset.dataset import MyDataset


def visualize_prediction_vs_ground_truth():
    # load config
    with open("conf/conf.yaml", "r") as f:
        cfg = Box(yaml.safe_load(f))
    # load dataset
    data = load_from_disk("data_merged")
    images = torch.tensor(data["observation.image"])
    joint = torch.tensor(data["observation.state"])
    action = torch.tensor(data["action"])

    dataset = MyDataset(
        action=action.numpy(),
        images=images.numpy(),
        joint=joint.numpy(),
        n_obs_steps=cfg.model.diffusion.n_obs_steps,
        horizon=cfg.model.diffusion.horizon,
    )

    sample = dataset[0]
    obs_img = sample["observation.image"].unsqueeze(0)
    obs_state = sample["observation.state"].unsqueeze(0)
    input_data = {"observation.image": obs_img, "observation.state": obs_state}

    # prepare model
    config = DiffusionConfig(
        n_obs_steps=cfg.model.diffusion.n_obs_steps,
        horizon=cfg.model.diffusion.horizon,
        n_action_steps=cfg.model.diffusion.n_action_steps,
        vision_backbone="resnet18",
        crop_shape=(48, 64),
    )

    class FeatureConfig:
        def __init__(self, feature_type, shape=None):
            self.type = feature_type
            self.shape = shape

    config.input_features = {
        "observation.image": FeatureConfig(FeatureType.VISUAL, shape=(3, 48, 64)),
        "observation.state": FeatureConfig(FeatureType.STATE, shape=(6,)),
    }
    config.output_features = {"action_is_pad": FeatureConfig(FeatureType.ACTION, shape=(6,))}

    policy = DiffusionPolicy(config)

    best_epoch = yaml.safe_load(f)["best_epoch"]
    weight_path = f"result/{cfg.wandb.train_name}/model/policy_epoch_{best_epoch}.safetensors"
    policy.load_state_dict(load_file(weight_path))
    policy.eval()

    with torch.no_grad():
        pred = policy.predict(input_data)  # shape: (1, horizon, 6)
    pred = pred[0].cpu().numpy()
    gt = sample["action"].numpy()

    # Plot
    fig, axs = plt.subplots(6, 1, figsize=(12, 10))
    for i in range(6):
        axs[i].plot(pred[:, i], label="Prediction")
        axs[i].plot(gt[:, i], "--", label="Ground Truth")
        axs[i].set_ylabel(f"Joint {i}")
        axs[i].legend()
    axs[-1].set_xlabel("Time step")
    plt.tight_layout()
    plt.savefig("result/prediction_vs_gt.png")
    plt.close()


# call the function
visualize_prediction_vs_ground_truth()
