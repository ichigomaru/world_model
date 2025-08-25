from tqdm import tqdm
import torch 
import wandb
from safetensors.torch import save_file
import os
import yaml

class TrainerSeparated:
    def __init__(self,
                policy,
                train_loader,
                val_loader,
                optimizer,
                epoch,
                device,
                save_path):
            
            self.policy = policy
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.optimizer = optimizer
            self.epoch = epoch
            self.device = device
            self.save_path = save_path
            
    def train(self):
        train_losses = []
        val_losses = []
        loss_min = 1.0e+10

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

                best_epoch_data = {'best_epoch': e}
                with open(os.path.join(self.save_path, 'best_epoch.yaml'), 'w') as f:
                    yaml.dump(best_epoch_data, f)

                save_file(self.policy.state_dict(), f"{self.save_path}/policy_latest.safetensors")

        return train_losses, val_losses


def forward(policy, batch, device):
    # バッチが辞書であることを仮定
    batch = {k: v.to(device) for k, v in batch.items()}
    policy.initialize(len(batch["action"]), device)
    loss, _ = policy(batch)
    return loss

def train_loop(policy, train_dataloader, optimizer, device):
    policy.train()
    total_loss = 0

    for i, batch in enumerate(train_dataloader):
        loss = forward(policy, batch, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / (i + 1)

def val_loop(policy, val_dataloader, device):
    policy.eval()
    total_loss = 0

    with torch.no_grad():
        for j, batch in enumerate(val_dataloader):
            loss = forward(policy, batch, device)
            total_loss += loss.item()

    return total_loss / (j + 1)