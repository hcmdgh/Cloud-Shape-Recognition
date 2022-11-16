import torch 
from torch import Tensor 
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms
from PIL import Image 
import csv 
import numpy as np 
import math 
from typing import Any, Dict, List, Tuple 
from tqdm import tqdm 
import cv2 

_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([224, 224]),
    transforms.ToTensor(), 
])


def prepare_dataset() -> np.ndarray:
    data_list = [] 
    
    with open('./train.csv', 'r', encoding='utf-8') as fp:
        reader = csv.DictReader(fp)
        
        for row in reader:
            filename = row['FileName']
            path = f'./images/{filename}'
            label = int(row['Code']) 
            
            data_list.append((path, label))

    data_arr = np.array(data_list, dtype=object)
    
    return data_arr 


def split_train_val_test_set(arr: np.ndarray, 
                             train_ratio: float,
                             val_ratio: float,
                             test_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert math.isclose(train_ratio + val_ratio + test_ratio, 1.) 
    N = len(arr) 
    perm = np.random.permutation(N)
    
    train_cnt = int(N * train_ratio)
    val_cnt = int(N * val_ratio)
    
    train_set = arr[perm[:train_cnt]]
    val_set = arr[perm[train_cnt : train_cnt + val_cnt]]
    test_set = arr[perm[train_cnt + val_cnt:]]
    assert len(train_set) + len(val_set) + len(test_set) == N  

    return train_set, val_set, test_set


class ImageDataset(Dataset):
    def __init__(self,
                 data_arr: np.ndarray):
        super().__init__()
        
        self.data_arr = data_arr 
        
    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        path, label = self.data_arr[index]
        
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = _transform(img)
        
        return img, label 
    
    def __len__(self) -> int:
        return len(self.data_arr)
    