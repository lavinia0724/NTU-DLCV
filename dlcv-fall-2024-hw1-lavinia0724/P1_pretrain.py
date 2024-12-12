import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from byol_pytorch import BYOL
import kornia.augmentation 

import warnings
warnings.filterwarnings("ignore")

# Data loader
class DLCVDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')])
        
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        img = Image.open(file)
        img = self.transform(img)
        # label = int(file.split("\\")[-1].split('_')[0]) # 因為檔名是 ex: "4_23.jpg" 也就是 "label_xx.jpg"
        
        return img


# hyper parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 350
batchSize = 64
learningRate = 2e-4
resnet = torchvision.models.resnet50(weights=None)

# BYOL augmentation function
augment_fn = nn.Sequential(
    kornia.augmentation.RandomHorizontalFlip(),
    kornia.augmentation.RandomAffine((-45, 45), p=1),
    kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.),
    kornia.filters.GaussianBlur2d((3, 3), (1.5, 1.5)),
    # kornia.RandomAffine(15),
)

# mode setting
learner = BYOL(
    resnet,
    image_size = 128,
    hidden_layer = 'avgpool'
)

learner = learner.to(device=device)
optimizer = torch.optim.Adam(learner.parameters() , lr=learningRate)


# data
trainDataset = DLCVDataset(".\\hw1_data\\p1_data\\mini\\train")
trainDataLoader = DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=True, num_workers=0, pin_memory=True)


# Training
bestTrainLoss = 10 # 一個很大的預設 loss
for epoch in range(epochs):
    learner.train()
    trainLoss = []


    for imgs in tqdm(trainDataLoader):
        imgs = imgs.to(device)

        loss = learner(imgs.to(device))

        optimizer.zero_grad()
        loss.backward() # 透過反向傳播獲得每個參數的梯度值
        optimizer.step() # 透過梯度下降執行參數更新

        learner.update_moving_average() # update moving average of target encoder

        trainLoss.append(loss.item())


    averageTrainLoss = sum(trainLoss) / len(trainLoss)
    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {averageTrainLoss:.4f}")


    if bestTrainLoss > averageTrainLoss:
        bestTrainLoss = averageTrainLoss
        torch.save(resnet.state_dict(), f"./DLCVHw1P1_PretrainBYOLModel.pt")
        print(f"New best model saved at epoch {epoch + 1}")  