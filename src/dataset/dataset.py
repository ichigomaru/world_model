import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, action, images, joint, sequence_length):
        self.action = torch.from_numpy(action).float()
        self.images = torch.from_numpy(images).float()
        self.joint = torch.from_numpy(joint).float()
        self.sequence_length = sequence_length
        
        self.images = (self.images / 127.5) - 1.0
        
    def __len__(self):
        return len(self.images) 

    def __getitem__(self, idx):
        image_seq = self.images[idx]
        action_seq = self.action[idx]
        joint_seq = self.joint[idx]
        
        return {"images": image_seq, "actions": action_seq, "joints": joint_seq}