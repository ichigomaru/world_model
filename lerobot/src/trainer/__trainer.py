# import torch
# from safetensors.torch import save_file
# from tqdm import tqdm

# import wandb


# class Trainer():
#     def __init__(self,
#                 vision_model,
#                 policy_model,
#                 train_dataloader,
#                 val_dataloader,
#                 optimizer,
#                 loss_fn,
#                 epoch,
#                 device,
#                 save_path):

#         self.vision = vision_model
#         self.policy = policy_model
#         self.train_dataloader = train_dataloader
#         self.val_dataloader = val_dataloader
#         self.optimizer = optimizer
#         self.loss_fn = loss_fn
#         self.epoch = epoch
#         self.device = device
#         self.save_path = save_path

#     def train_model(self):
#         train_loss_log = []
#         validation_loss_log = []
#         loss_min = 1.0e+10

#         for e in tqdm(range(self.epoch)):
#             train_loss = train_loop(self.vision, self.policy, self.train_dataloader, self.optimizer, self.loss_fn, self.device)
#             val_loss = validation_loop(self.vision, self.policy, self.val_dataloader, self.loss_fn, self.device)

#             train_loss_log.append(train_loss)
#             validation_loss_log.append(val_loss)

#             wandb.log({
#                 'train_loss': train_loss,
#                 'validation_loss': val_loss,
#                 'epoch': e,
#             })

#             if val_loss < loss_min:
#                 # VisionとPolicyをそれぞれ保存
#                 save_file(self.vision.state_dict(), f'{self.save_path}/best_vision.safetensors')
#                 save_file(self.policy.state_dict(), f'{self.save_path}/best_policy.safetensors')
#                 loss_min = val_loss

#         return train_loss_log, validation_loss_log


# def forward_model(vision, policy, images, follower, goal_onehot, device):
#     B, T, C, H, W = images.shape
#     x = images.view(B * T, C, H, W).to(device)
#     vision_out = vision(x)
#     vision_out = vision_out.view(B, T, -1)[:, :-1]
#     follower_input = follower[:, :-1].to(device)
#     goal_input = goal_onehot[:, :-1].to(device)     # (B, T-1, 2)

#     policy_input = torch.cat([vision_out, follower_input, goal_input], dim=-1)
#     policy.initialize(B, device)
#     prediction = policy(policy_input)
#     return prediction


# def train_loop(vision, policy, train_dataloader, optimizer, loss_fn, device):
#     vision.train()
#     policy.train()
#     total_loss = 0
#     for i, (images, follower, goal_onehot, leader) in enumerate(train_dataloader):
#         images, follower, goal_onehot, leader = images.to(device), follower.to(device), goal_onehot.to(device), leader.to(device)

#         prediction = forward_model(vision, policy, images, follower, goal_onehot, device)
#         target = leader[:, 1:]  # 予測対象は次時刻のリーダー状態

#         loss = loss_fn(prediction, target)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#     return total_loss / (i + 1)


# def validation_loop(vision, policy, val_dataloader, loss_fn, device):
#     vision.eval()
#     policy.eval()
#     total_loss = 0
#     with torch.no_grad():
#         for j, (images, follower, goal_onehot, leader) in enumerate(val_dataloader):
#             images, follower, goal_onehot, leader = images.to(device), follower.to(device), goal_onehot.to(device), leader.to(device)

#             prediction = forward_model(vision, policy, images, follower, goal_onehot, device)
#             target = leader[:, 1:]

#             loss = loss_fn(prediction, target)
#             total_loss += loss.item()
#     return total_loss / (j + 1)
