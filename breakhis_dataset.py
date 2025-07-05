import os
import torch
import pandas as pd
#from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image


class BreakhisDataset(Dataset):
    def __init__(self, list_dataset, transform=None):
        """
        list_dataset = [(img_path_1, label_1), ...]
        """
        self.list_dataset = list_dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.list_dataset)
    
    def __getitem__(self, index):
        
        img_name = self.list_dataset[index][0]
        img_label = self.list_dataset[index][1]
        
        img = Image.open(img_name)
        img = self.transform(img)
        
        return img, img_label
        
        
        