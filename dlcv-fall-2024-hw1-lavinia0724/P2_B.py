import os
import random
import torch
import torchvision
from torchvision import models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
# from torchvision.models.segmentation.deeplabv3 import DeepLabHead, FCNHead

import mean_iou_evaluate

import warnings
warnings.filterwarnings("ignore")



class DLCVDataset(Dataset):
    def __init__(self, path, transform=None, mode='train'):
        assert mode in ['train', 'validation']

        self.path = path
        self.images = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')])
        # self.labels = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png')])
        self.mode = mode

        self.labels = [] # because test don't use labels
        if self.mode == "train":
            self.labels = mean_iou_evaluate.read_masks(path)

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        imgFile = self.images[idx]
        img = Image.open(imgFile)

        # label = self.labels[idx]
        # label = Image.open(labelFile)

        # Convert label (NumPy array) to PIL image
        # label = Image.fromarray(label)
        if self.mode == "train":
            label = Image.fromarray(self.labels[idx])

            if random.random() > 0.5:
                img = transforms.functional.hflip(img)
                label = transforms.functional.hflip(label)
            if random.random() > 0.5:
                img = transforms.functional.vflip(img)
                label = transforms.functional.vflip(label)

            if self.transform:
                img = self.transform(img)

            label = np.array(label)
            return img, label
        else:
            if self.transform:
                img = self.transform(img)
            return img


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 平衡因子，用于控制不同类的权重
        self.gamma = gamma  # 调节因子，用于降低对易分类样本的关注
        self.ce_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=6)  # 基于 CrossEntropy 的计算

    def forward(self, predict, target):
        logpt = -self.ce_loss(predict, target)  # 计算交叉熵损失
        pt = torch.exp(logpt)  # 获得预测概率

        # 计算 Focal Loss，其中 alpha 和 gamma 为调节因子
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * (-logpt)
        return focal_loss.mean() 


# preprocessing transform
transform = transforms.Compose([
    # transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 100
batchSize = 8
learningRate = 1e-4
numClass = 7


# Data
trainDataset = DLCVDataset("./hw1_data/p2_data/train/", transform=transform, mode='train')
trainDataLoader = DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=True, num_workers=0, pin_memory=True)

validationDataset = DLCVDataset("./hw1_data/p2_data/validation/", transform=transform, mode='train')
validationDataLoader = DataLoader(dataset=validationDataset, batch_size=batchSize, shuffle=True, num_workers=0, pin_memory=True)


# model = torchvision.models.mobilenet_v3_large(weights='DEFAULT')
# model.classifier = nn.Sequential(
#     nn.Conv2d(960, 7, kernel_size=1),  # Replace 7 with your number of classes
#     nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False)  # Upsample to the original image size
# )

# model = torchvision.models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
# model.classifier = DeepLabHead(2048, 7)

model = models.segmentation.deeplabv3_resnet50(num_classes=numClass, weight='DEFAULT')
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learningRate)
# optimizer = torch.optim.SGD(model.parameters(), momentum = 0.85, lr=learningRate, weight_decay=5e-4)


# Scheduler Initialization
# scheduler = torch.optim.lr_scheduler.OneCycleLR(
#     optimizer, max_lr=1e-4, steps_per_epoch=len(trainDataLoader), epochs=epochs
# )


# criterion = nn.CrossEntropyLoss()
# criterion = nn.CrossEntropyLoss(ignore_index = 6)
criterion = FocalLoss()


savePredictedEpochs = [0, 49, 99]  # 早期、中期和後期

# Training
if __name__ == '__main__':
    bestValidationmIOU = 0.0  # Initialize best validation mIoU
    for epoch in range(epochs):
        model.train()
        trainLoss = []
        # trainIOU = []
        trainCorrect = 0
        totalLabels = 0
        predictList = []
        labelsList = []

        for imgs, labels in tqdm(trainDataLoader):
            labels = labels.long()
            imgs, labels = imgs.to(device), labels.to(device)

            predict = model(imgs)['out'] 
            # predict = F.interpolate(predict, size=(labels.size(1), labels.size(2)), mode='bilinear', align_corners=False)  # Upsample

            # labels = labels.long()
            loss = criterion(predict, labels)

            optimizer.zero_grad()
            loss.backward()  # Perform backpropagation

            # grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            
            optimizer.step()  # Update parameters

            trainLoss.append(loss.item())

            # _, predicted = predict.max(1)
            # predictList.append(predicted.detach().cpu().numpy().astype(np.int64))
            # labelsList.append(labels.detach().cpu().numpy().astype(np.int64))
            # trainIOU.append(mean_iou_evaluate.mean_iou_score(predictTrans, labelsTrans))

            # trainCorrect += (predict.argmax(dim=1) == labels).sum().item()
            # totalLabels += labels.size(0) * labels.size(1) * labels.size(2)  # Total pixels

        averageTrainLoss = sum(trainLoss) / len(trainLoss)
        # iouScore = mean_iou_evaluate.mean_iou_score(np.concatenate(predictList, axis=0), np.concatenate(labelsList, axis=0))
        # trainAccuracy = trainCorrect / totalLabels


        # print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {averageTrainLoss:.4f}, Train Accuracy: {trainAccuracy * 100:.2f}%, mIoU Score: {iouScore:.5f}")
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {averageTrainLoss:.4f}")

        # Validation --------------------------------------------------------------------
        model.eval()
        validationLoss = []
        # validationIOU = []
        validationCorrect = 0
        totalLabels = 0
        predictList = []
        labelsList = []

        with torch.no_grad():
            for imgs, labels in validationDataLoader:
                labels = labels.long()
                imgs, labels = imgs.to(device), labels.to(device)

                predict = model(imgs)['out'] 
                # predict = F.interpolate(predict, size=(labels.size(1), labels.size(2)), mode='bilinear', align_corners=False)  # Upsample
                
                loss = criterion(predict, labels)

                # predictTrans = predict.argmax(dim=1).cpu().numpy().astype(np.int64)
                # labelsTrans = labels.cpu().numpy().astype(np.int64)

                validationLoss.append(loss.item())

                _, predicted = predict.max(1)
                predictList.append(predicted.detach().cpu().numpy().astype(np.int64))
                labelsList.append(labels.detach().cpu().numpy().astype(np.int64))
                
                # trainIOU.append(mean_iou_evaluate.mean_iou_score(predictTrans, labelsTrans))

                # predictList.append(torch.argmax(predict, dim=1).detach().cpu().numpy().astype(np.int64))
                # labelsList.append(labels.detach().cpu().numpy().astype(np.int64))

                # validationCorrect += (predict.argmax(dim=1) == labels).sum().item()
                # totalLabels += labels.size(0) * labels.size(1) * labels.size(2)  # Total pixels

            averageValidationLoss = sum(validationLoss) / len(validationLoss)
            iouScore = mean_iou_evaluate.mean_iou_score(np.concatenate(predictList, axis=0), np.concatenate(labelsList, axis=0))
            # validationAccuracy = validationCorrect / totalLabels

            # print(f"Validation Loss: {averageValidationLoss:.4f}, Validation Accuracy: {validationAccuracy * 100:.2f}%, mIoU Score: {iouScore:.5f}")
            print(f"Validation Loss: {averageValidationLoss:.4f}, mIoU Score: {iouScore:.5f}")


            if bestValidationmIOU < iouScore:
                bestValidationmIOU = iouScore
                torch.save(model.state_dict(), f"DLCVHw1P2B_BestModel_Epoch{epoch + 1}.ckpt")

                print(f"New best validation model saved at epoch {epoch + 1}")


            if epoch in savePredictedEpochs:
                torch.save(model.state_dict(), f"DLCVHw1P2B_Epoch{epoch + 1}.ckpt")

        # scheduler.step()


    print(f"Best mIoU Score: {bestValidationmIOU:.5f}")
