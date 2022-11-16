import torch 
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms
from PIL import Image 
import csv 
import numpy as np 
import math 
from typing import Any, Dict, List, Tuple 
from tqdm import tqdm 


def load_full_dataset() -> np.ndarray:
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(), 
    ])
    
    data_list = [] 
    
    with open('./train.csv', 'r', encoding='utf-8') as fp:
        reader = csv.DictReader(fp)
        
        for row in tqdm(list(reader)):
            filename = row['FileName']
            path = f'./images/{filename}'
            label = int(row['Code']) 
            
            img = Image.open(path)

            img = transform(img)
            
            if img.shape != (3, 224, 224):
                print(path)
                print(img.shape) 

            data_list.append(
                dict(
                    img = img, 
                    label = label,  
                )
            )

    data_list = np.array(data_list, dtype=object)
    
    return data_list 


def split_train_val_test_set(data_list: np.ndarray, 
                             train_ratio: float,
                             val_ratio: float,
                             test_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert math.isclose(train_ratio + val_ratio + test_ratio, 1.) 
    N = len(data_list) 
    perm = np.random.permutation(N)
    
    train_cnt = int(N * train_ratio)
    val_cnt = int(N * val_ratio)
    
    train_set = data_list[perm[:train_cnt]]
    val_set = data_list[perm[train_cnt : train_cnt + val_cnt]]
    test_set = data_list[perm[train_cnt + val_cnt:]]
    assert len(train_set) + len(val_set) + len(test_set) == N  

    return train_set, val_set, test_set
