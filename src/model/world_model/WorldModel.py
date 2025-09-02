import torch
import torch.nn as nn

class WorldModel(nn.Module):
    def __init__(self, vision_encoder, rssm, vision_decoder):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.rssm = rssm
        self.vision_decoder = vision_decoder

    def forward(self, images, actions):

        b, t, c, h, w = images.shape
        embedded_obs = self.vision_encoder(images.view(b * t, c, h, w))
        embedded_obs = embedded_obs.view(b, t, -1)
        
        deterministic_states, stochastic_states, prior_dists, posterior_dists = self.rssm(
            embedded_obs, actions
        )
        
        latent_states_for_decoder = torch.cat([deterministic_states, stochastic_states], dim=-1)
        
        reconstructed_images = self.vision_decoder(
            latent_states_for_decoder.view(b * t, -1)
        )
        
        reconstructed_images = reconstructed_images.view(b, t, c, h, w)
        
        return reconstructed_images, deterministic_states, stochastic_states, prior_dists, posterior_dists