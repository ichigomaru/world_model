import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

class ObsTransform:
    def __init__(self, height, width, do_aug):
        if do_aug:
            self.obs_transform = A.Compose(
            [      
            A.Resize(height=height, width=width, p=1.0),
            A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.1),
            A.RandomBrightnessContrast(p=0.3),
            A.MotionBlur(blur_limit=(3, 11), p=0.1),
            A.GaussNoise(var_limit=(50, 300), p=0.3),
            A.ISONoise(intensity=(0.5,1.5), p=0.3),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), p=1.0),
            ToTensorV2(),
            ]   
            )
        else:
            self.obs_transform = A.Compose(
            [
            A.Resize(height=height, width=width, p=1.0),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), p=1.0),
            ToTensorV2(),
            ]   
            )

    def __call__(self, obs):
        if obs.ndim == 5:
            nobs = []
            for t in range(obs.shape[0]):
                for b in range(obs.shape[1]):
                    img = obs[t][b].permute(2, 0, 1) # (H, W, C) -> (C, H, W)
                    _nobs = self.obs_trainsform(image=img)['image'] #(C, H, W)
                    #_nobs = self.obs_transform(image=obs[t])['image']
                    nobs.append(_nobs.unsqueeze(0)) #(1, C, H, W)
                nobs = torch.cat(nobs, 0) # (T * B, C, H, W)
            #データ数、シークエンス分、height、width、RGBで構成されているから、入れ子構造で全部通すようにする。現状はタイムステップしか考慮されていない。
        elif obs.ndim == 3:
            nobs = self.obs_transform(image=obs)['image'].unsqueeze(0)
        else:
            raise ValueError(f"Invalid obs shape = {obs.shape}")
        
        return nobs