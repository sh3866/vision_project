from __future__ import print_function, division
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import os
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter

# CSV 파일 경로
train_csv_path = '/home/pink/intern/project_el/datasets/classification/train/re_data_df_first.csv'
valid_csv_path = '/home/pink/intern/project_el/datasets/classification/valid/re_data_df_first.csv'
test_csv_path = '/home/pink/intern/project_el/datasets/classification/test/re_data_df_first.csv'

# 이미지 파일 경로
img_file_path = '/home/pink/intern/project_el/datasets/first'

# 기본 이미지 어떻게 transform 할것인지 transform 정의
image_transform = transforms.Compose([
    transforms.Resize((977, 459), antialias=True)
    #,transforms.ToTensor()
])

# 데이터 증강 정의
image_augmentating = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop((977, 459), antialias=True),
])


class CustomDataset(Dataset):
    def __init__(self, csv_file, img_file, transform=None, target_transform=None, data_augmentation=None):
        
        # csv file에서 filename과 label_re column만 가져옴
        self.img_labels = pd.read_csv(csv_file, usecols=["filename", "label", "label_re"])
        
        # 나머지는 매개변수 그대로 입력받음
        self.img_file = img_file
        self.transform = transform
        self.target_transform = target_transform
        self.data_augmentation = data_augmentation
        
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        
        # os.path.join을 사용하여 경로결합하여 정확한 그 인덱스의 이미지 img_path에 저장
        # img_path = os.path.join(self.img_file, self.img_labels.iloc[idx, 0])
        
        
        # 0,1일때 사건 나누어서 분류
        img_filename = self.img_labels.iloc[idx, 0]
        label_for_image = self.img_labels.iloc[idx, 1]

        if label_for_image == 0:
            subfolder = 'non_fault' 
        else:
            subfolder = 'fault'

        img_path = os.path.join(self.img_file, subfolder, img_filename)
        
        # read_image를 사용하여 이미지를 텐서로 변환 + float 추가
        image = read_image(img_path).float()

        
        # label 값도 해당 인덱스 찾아서 저장
        label = self.img_labels.iloc[idx, 2]
        
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
            
        if self.data_augmentation:
            image = self.data_augmentation(image)
            
        return image, label


# Dataloader 생성

def make_data_loader(args):
    # CustomDataset 바탕으로 train_dataset 생성
    train_dataset = CustomDataset(csv_file=train_csv_path, img_file=img_file_path, transform=image_transform, target_transform=None, data_augmentation=None)
    test_dataset = CustomDataset(csv_file=test_csv_path, img_file=img_file_path, transform=image_transform, target_transform=None, data_augmentation=None)
    
    # Get Dataloader
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader

def valid_data_loader(args):
    valid_dataset = CustomDataset(csv_file=valid_csv_path, img_file=img_file_path, transform=image_transform, target_transform=None, data_augmentation=None)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)
    
    return valid_loader