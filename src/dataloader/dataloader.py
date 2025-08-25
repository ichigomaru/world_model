import torch
from torch.utils.data import Dataset, DataLoader, random_split

class MyDataloader():
    def __init__(self,
                mydataset: Dataset,
                ratio: list,
                batch_size: int,
                seed=0):
        
        self.mydataset = mydataset
        self.ratio = ratio
        self.batch_size = batch_size
        self.seed = seed

    def prepare_data(self):
        generator = torch.Generator().manual_seed(self.seed)
        train_data, val_data, test_data = random_split(self.mydataset, lengths=self.ratio, generator=generator)

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_data, batch_size=self.batch_size)
        test_loader = DataLoader(test_data, batch_size=self.batch_size)

        return train_loader, val_loader, test_loader