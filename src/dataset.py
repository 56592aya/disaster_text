import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from torch.util.data import Dataset
from torch.util.data import DataLoader


# Map-style datasets
# A map-style dataset is one that implements the __getitem__() and __len__() protocols, and represents a map from (possibly non-integral) indices/keys to data samples.
# For example, such a dataset, when accessed with dataset[idx], could read the idx-th image and its corresponding label from a folder on the disk.


class DisasterDataset(Dataset):
    """Dataset for Disaster Tweets"""
    def __init__(self, tweets, target):
        super(DisasterDataset, self).__init__()
        self.tweets = tweets
        self.target = targets
    
    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        tweet = self.tweets[idx]
        target = self.targets[idx]
        sample = {
            'tweets':torch.tensor(tweet, dtype=torch.long),
            'targets':torch.tensor(target, dtype=torch.float)
            }
        return sample
