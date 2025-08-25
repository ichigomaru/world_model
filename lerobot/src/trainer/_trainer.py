# import os

# import torch
# from safetensors.torch import save_file
# from tqdm import tqdm

# import wandb


# class TrainerSeparated:
#     def __init__(self,
#                 vision,
#                 policy,
#                 train_loader,
#                 val_loader,
#                 optimizer,
#                 loss_fn,
#                 epoch,
#                 device,
#                 save_path):

#             self.vision = vision
#             self.policy = policy
#             self.train_loader = train_loader
#             self.val_loader = val_loader
#             self.optimizer = optimizer
#             self.loss_fn = loss_fn
#             self.epoch = epoch
#             self.device = device
#             self.save_path = save_path

#     def train(self):
#         train_losses = []
#         val_losses = []
#         loss_min = 1.0e+10

#         os.makedirs(self.save_path, exist_ok=True)

#         for e in tqdm(range(self.epoch)):
#             train_loss = train_loop(self.vision, self.policy, self.train_loader, self.optimizer, self.loss_fn, self.device)
#             val_loss = val_loop(self.vision, self.policy, self.val_loader, self.loss_fn, self.device)

#             train_losses.append(train_loss)
#             val_losses.append(val_loss)

#             wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": e})

#             if val_loss < loss_min:
#                 save_file(self.vision.state_dict(), f"{self.save_path}/vision_epoch_{e}.safetensors")
#                 save_file(self.policy.state_dict(), f"{self.save_path}/policy_epoch_{e}.safetensors")
#                 loss_min = val_loss

#                 # best_epoch.yaml の保存
#                 best_epoch_data = {'best_epoch': e}
#                 with open(os.path.join(self.save_path, 'best_epoch.yaml'), 'w') as f:
#                     import yaml
#                     yaml.dump(best_epoch_data, f)

#                 # 最終エポックのモデルも保存
#                 save_file(self.vision.state_dict(), f"{self.save_path}/vision_latest.safetensors")
#                 save_file(self.policy.state_dict(), f"{self.save_path}/policy_latest.safetensors")

#         return train_losses, val_losses


# def forward(vision, policy, images, joint, device):
#     #print("B, T, C, H, W = images.shape直前のimages.shape,", images.shape)
#     B, T, C, H, W = images.shape
#     images = images.view(B * T, C, H, W).to(device)

#     x = vision(images)
#     x = x.view(B, T, -1)[:, :-1]
#     joint = joint[:, :-1].to(device)

#     policy_input = torch.cat((x, joint), dim=-1)
#     policy.initialize(B, device)
#     prediction = policy(policy_input)
#     return prediction

# def train_loop(vision, policy, train_dataloader, optimizer, loss_fn, device):
#     vision.train()
#     policy.train()
#     total_loss = 0


#     for i, (images, joint, action) in enumerate(train_dataloader):
#         images, joint, action = images.to(device), joint.to(device), action.to(device)

#         prediction = forward(vision, policy, images, joint, device)
#         target = action[:, 1:]

#         loss = loss_fn(prediction, target)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     return total_loss / (i + 1)

# def val_loop(vision, policy, val_dataloader, loss_fn, device):
#     vision.eval()
#     policy.eval()
#     total_loss = 0

#     with torch.no_grad():
#         for j, (images, joint, action) in enumerate(val_dataloader):
#             images, joint, action = images.to(device), joint.to(device), action.to(device)

#             prediction = forward(vision, policy, images, joint, device)
#             target = action[:, 1:]

#             loss = loss_fn(prediction, target)
#             total_loss += loss.item()

#     return total_loss / (j + 1)
