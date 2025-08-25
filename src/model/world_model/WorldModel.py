import torch
import torch.nn as nn

class WorldModel(nn.Module):
    """
    リファクタリングされたRSSMと連携するWorldModelクラス。
    """
    def __init__(self, vision_encoder, rssm, vision_decoder):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.rssm = rssm
        self.vision_decoder = vision_decoder

    def forward(self, images, actions):
        """
        学習時のフォワードパス。シーケンス全体を一度に処理する。
        """
        # 1. Encoder: 画像シーケンスを潜在変数シーケンスに変換
        b, t, c, h, w = images.shape
        embedded_obs = self.vision_encoder(images.view(b * t, c, h, w))
        embedded_obs = embedded_obs.view(b, t, -1)
        
        # 2. RSSM: 状態シーケンスを予測
        deterministic_states, stochastic_states, prior_dists, posterior_dists = self.rssm(
            embedded_obs, actions
        )
        
        # 3. Decoder: 状態シーケンスから画像シーケンスを復元
        reconstructed_images = self.vision_decoder(
            stochastic_states.view(b * t, -1)
        )
        reconstructed_images = reconstructed_images.view(b, t, c, h, w)
        
        return reconstructed_images, deterministic_states, stochastic_states, prior_dists, posterior_dists
