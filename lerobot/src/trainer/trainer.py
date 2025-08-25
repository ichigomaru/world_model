import os

import torch
import wandb
import yaml
from safetensors.torch import save_file
from tqdm import tqdm


class TrainerSeparated:
    def __init__(self, policy, train_loader, val_loader, optimizer, epoch, device, save_path):
        self.policy = policy
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.epoch = epoch
        self.device = device
        self.save_path = save_path

    def train(self):  # train_loopとval_loopを呼ぶ
        train_losses = []
        val_losses = []
        loss_min = 1.0e10

        os.makedirs(self.save_path, exist_ok=True)

        for e in tqdm(range(self.epoch)):
            train_loss = train_loop(self.policy, self.train_loader, self.optimizer, self.device)
            val_loss = val_loop(self.policy, self.val_loader, self.device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": e})

            if val_loss < loss_min:
                save_file(self.policy.state_dict(), f"{self.save_path}/policy_epoch_{e}.safetensors")
                loss_min = val_loss

                best_epoch_data = {"best_epoch": e}
                with open(os.path.join(self.save_path, "best_epoch.yaml"), "w") as f:
                    yaml.dump(best_epoch_data, f)

                # save_file(vision_encoder.state_dict(), f"{self.save_path}/vision_epoch_{e}.safetensors")
                save_file(self.policy.state_dict(), f"{self.save_path}/policy_latest.safetensors")

        return train_losses, val_losses


# def forward(policy, batch, device):
#     batch = {k: v.to(device) for k, v in batch.items()}
#     loss, _ = policy.forward(batch)
#     return loss


def train_loop(policy, train_dataloader, optimizer, device):
    # for name, param in policy.named_parameters():
    #     if param.requires_grad and "vision_encoder" in name:
    #         print(f"Trainable vision param: {name}")
    policy.train()
    total_loss = 0

    for i, batch in enumerate(train_dataloader):
        # t_start = time.time()

        batch = {k: v.to(device) for k, v in batch.items()}
        # t_data_loaded = time.time()

        optimizer.zero_grad()
        loss, _ = policy.forward(batch)
        # t_model_forward = time.time()
        # print(f"Step {i}, loss: {loss.item()}")
        loss.backward()
        optimizer.step()
        # t_backprop = time.time()
        # print(f"[Batch {i}]")
        # print(f"  Data loading:     {t_data_loaded - t_start:.3f} sec")
        # print(f"  Model forward:    {t_model_forward - t_data_loaded:.3f} sec")
        # print(f"  Backprop + opt:   {t_backprop - t_model_forward:.3f} sec")
        # print(f"  Total batch time: {t_backprop - t_start:.3f} sec")
        total_loss += loss.item()

    return total_loss / (i + 1)


def val_loop(policy, val_dataloader, device):
    policy.eval()
    total_loss = 0

    with torch.no_grad():
        for j, batch in enumerate(val_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, _ = policy.forward(batch)
            # print(f"Validation batch {j}, loss: {loss.item()}")
            total_loss += loss.item()

    return total_loss / (j + 1)
