from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal, kl_divergence

@dataclass
class LossParameters:
    origin_image: Tensor
    recon_image: Tensor
    prior_dist: Normal
    posterior_dist: Normal
    kl_balance: float
    kl_beta: float

def world_model_loss(params: LossParameters):
    # 1. Reconstruction Loss
    #    単純な平均二乗誤差 (MSE) を計算
    recon_loss = F.mse_loss(params.recon_image, params.origin_image)

    # 2. KL損失 (KL Balancing Loss)
    #    priorとposteriorの勾配を分離して計算
    
    # a) posteriorを更新するための損失
    #    priorの勾配は止める
    prior_dist_detached = Normal(params.prior_dist.mean.detach(), params.prior_dist.scale.detach())
    posterior_loss = kl_divergence(params.posterior_dist, prior_dist_detached)

    # b) priorを更新するための損失
    #    posteriorの勾配は止める
    posterior_dist_detached = Normal(params.posterior_dist.mean.detach(), params.posterior_dist.scale.detach())
    prior_loss = kl_divergence(posterior_dist_detached, params.prior_dist)

    kl_loss = params.kl_balance * posterior_loss.mean() + (1 - params.kl_balance) * prior_loss.mean()

    total_loss = recon_loss + params.kl_beta * kl_loss

    loss_dict = {
        "total_loss": total_loss,
        "reconstruction_loss": recon_loss,
        "kl_loss": kl_loss,
    }

    return loss_dict