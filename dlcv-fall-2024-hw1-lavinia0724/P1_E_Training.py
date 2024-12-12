import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import RandomApply
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from byol_pytorch import BYOL
# import kornia.augmentation 

import warnings
warnings.filterwarnings("ignore")

# Data loader
class DLCVDataset(Dataset):
    def __init__(self, path, mode='train'):
        assert mode in ['train', 'validation']

        self.path = path
        self.mode = mode
        self.files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')])

        self.trainTransform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((128, 128)),
            transforms.RandomGrayscale(p=0.3),
            RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),  
                         transforms.GaussianBlur((3, 3), (1.0, 2.0))], p=0.2),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.validationTransform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        img = Image.open(file)

        if self.mode == "train":
            img = self.trainTransform(img)
        else:
            img = self.validationTransform(img)

        label = int(file.split("/")[-1].split('_')[0]) # 因為檔名是 ex: "4_23.jpg" 也就是 "label_xx.jpg"
        
        return img, label


# hyper parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 100
batchSize = 256
learningRate = 1e-3


model = torchvision.models.resnet50(weights=None)
model.load_state_dict(torch.load("./DLCVHw1P1_PretrainBYOLModel.pt"))  # 載入 pretrain model
model.fc = torch.nn.Linear(2048, 65)

for param in model.parameters():
    param.requires_grad = False
    
for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(device)


optimizer = torch.optim.Adam(model.parameters() , lr=learningRate, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()


# data
trainDataset = DLCVDataset("./hw1_data/p1_data/office/train/", mode='train')
trainDataLoader = DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=True, num_workers=0, pin_memory=True)

validationDataset = DLCVDataset("./hw1_data/p1_data/office/val/", mode='validation')
validationDataLoader = DataLoader(dataset=validationDataset, batch_size=batchSize, shuffle=True, num_workers=0, pin_memory=True)

# Training
if __name__ == '__main__':
    bestValidationAccuracy = 0.0 # 一個很小的預設 Accuracy
    for epoch in range(epochs):
        model.train()
        trainLoss = []
        trainCorrect = 0
        totalLabels = 0

        for imgs, labels in tqdm(trainDataLoader):
            imgs, labels = imgs.to(device), labels.to(device)
            
            predict = model(imgs)
            loss = criterion(predict, labels)

            optimizer.zero_grad()
            loss.backward() # 透過反向傳播獲得每個參數的梯度值
            optimizer.step() # 透過梯度下降執行參數更新

            _, predictedLabel = torch.max(predict, 1)
            trainCorrect += predictedLabel.eq(labels).sum().item()
            totalLabels += labels.size(0)

            trainLoss.append(loss.item())


        averageTrainLoss = sum(trainLoss) / len(trainLoss)
        trainAccuracy = trainCorrect / totalLabels

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {averageTrainLoss:.4f}, Train Accuracy: {trainAccuracy * 100:.2f}%")


        # Validation --------------------------------------------------------------------
        model.eval()
        validationLoss = []
        validationCorrect = 0
        totalLabels = 0

        with torch.no_grad():
            for imgs, labels in validationDataLoader:
                imgs, labels = imgs.to(device), labels.to(device)
                
                predict = model(imgs)
                loss = criterion(predict, labels)

                _, predictedLabel = torch.max(predict, 1)
                validationCorrect += predictedLabel.eq(labels).sum().item()
                totalLabels += labels.size(0)

                validationLoss.append(loss.item())

            averageValidationLoss = sum(validationLoss) / len(validationLoss)
            validationAccuracy = validationCorrect / totalLabels

            print(f"Validation Loss: {averageValidationLoss:.4f}, Validation Accuracy: {validationAccuracy * 100:.2f}%")


            if bestValidationAccuracy < validationAccuracy:
                bestValidationAccuracy = validationAccuracy
                torch.save(model.state_dict(), f"./DLCVHw1P1E_BestModel.ckpt")
                
                print(f"New best validation model saved at epoch {epoch + 1}")  


    print(f"Best Validation Accuracy: {bestValidationAccuracy * 100:.2f}%")