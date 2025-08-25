import os
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence, Normal, Independent
import wandb
from safetensors.torch import save_file

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, epoch, device, kl_loss_scale, save_path):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.epochs = epoch
        self.device = device
        self.kl_loss_scale = kl_loss_scale
        self.save_path = save_path
        self.best_val_loss = float('inf')
        os.makedirs(self.save_path, exist_ok=True)

    def _calculate_loss(self, images, actions):

        recon_images, _, _, priors, posteriors = self.model(images, actions)
        
        loss_reconstruction = F.mse_loss(recon_images, images)
        
        kl_loss = 0
        for prior_dist, posterior_dist in zip(priors, posteriors):
            kl_loss += kl_divergence(
                Independent(posterior_dist, 1), Independent(prior_dist, 1)
            ).mean()
        
        kl_loss /= len(priors)
        
        total_loss = loss_reconstruction + kl_loss

        return {
            "total_loss": total_loss,
            "reconstruction_loss": loss_reconstruction,
            "kl_loss": kl_loss
        }

    def _run_epoch(self, data_loader, is_training):

        total_losses = {"total_loss": 0, "reconstruction_loss": 0, "kl_loss": 0}
        
        progress_bar = tqdm(data_loader, desc="Training" if is_training else "Validation")

        for batch in progress_bar:
            images = batch["images"].to(self.device)
            actions = batch["actions"].to(self.device)
            
            if is_training:
                self.optimizer.zero_grad()
                losses = self._calculate_loss(images, actions)

                losses["total_loss"].backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    losses = self._calculate_loss(images, actions)
            
            for key, value in losses.items():
                total_losses[key] += value.item()
            
            progress_bar.set_postfix(loss=losses["total_loss"].item())

        avg_losses = {key: value / len(data_loader) for key, value in total_losses.items()}
        return avg_losses

    def train(self):

        for e in range(self.epochs):
            print(f"\n--- Epoch {e+1}/{self.epochs} ---")
            
            self.model.train()
            train_losses = self._run_epoch(self.train_loader, is_training=True)
            
            self.model.eval()
            val_losses = self._run_epoch(self.val_loader, is_training=False)

            print(f"Epoch {e+1}: Train Loss: {train_losses['total_loss']:.4f}, Val Loss: {val_losses['total_loss']:.4f}")
            
            wandb.log({
                "epoch": e,
                "train/total_loss": train_losses['total_loss'],
                "train/reconstruction_loss": train_losses['reconstruction_loss'],
                "train/kl_loss": train_losses['kl_loss'],
                "val/total_loss": val_losses['total_loss'],
                "val/reconstruction_loss": val_losses['reconstruction_loss'],
                "val/kl_loss": val_losses['kl_loss'],
            })

            if val_losses['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['total_loss']
                best_model_path = os.path.join(self.save_path, "world_model_best.safetensors")
                save_file(self.model.state_dict(), best_model_path)
                print(f"New best model saved to {best_model_path} with validation loss {val_losses['total_loss']:.4f}")

                best_epoch_data = {'best_epoch': e, 'val_loss': val_losses['total_loss']}
                with open(os.path.join(self.save_path, 'best_epoch.yaml'), 'w') as f:
                    yaml.dump(best_epoch_data, f)
