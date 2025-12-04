import torch
from torch.utils.data import Dataset


class ASLDataSet(Dataset):
    def __init__(self, x_df, y_df, device):
        self.xs = torch.tensor(x_df).float().to(device)
        self.ys = torch.tensor(y_df).to(device)
        self.device = device

        assert len(self.xs) == len(self.ys)

    def __getitem__(self, idx):
        x = self.xs[idx]
        y = self.ys[idx]
        return x, y

    def __len__(self):
        return len(self.xs)

    def save(self, path):
        data = {"xs": self.xs.cpu(), "ys": self.ys.cpu()}
        torch.save(data, path)

    @classmethod
    def load(cls, path, device=None):
        data = torch.load(path, map_location=device or "cpu")
        return cls(data["xs"], data["ys"], device=device)
