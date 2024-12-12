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
import sys
import imageio.v2 as imageio

import warnings
warnings.filterwarnings("ignore")


cls_color = {
    0:  [0, 255, 255],
    1:  [255, 255, 0],
    2:  [255, 0, 255],
    3:  [0, 255, 0],
    4:  [0, 0, 255],
    5:  [255, 255, 255],
    6: [0, 0, 0],
}

class DLCVDataset(Dataset):
    def __init__(self, path, transform=None):

        self.path = path
        self.images = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')])

        self.transform = transform

        self.savedImgs = sorted([os.path.splitext(os.path.basename(f))[0].split("_")[0] + '_mask.png' for f in self.images])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        imgFile = self.images[idx]
        img = Image.open(imgFile)

        img = self.transform(img)
    
        return img, self.savedImgs[idx]



# preprocessing transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


testing_images_directory = sys.argv[1]
output_images_directory = sys.argv[2]

if not os.path.exists(output_images_directory):
    os.mkdir(output_images_directory)


# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batchSize = 8
numClass = 7


# Data
testDataset = DLCVDataset(testing_images_directory, transform=transform)
testDataLoader = DataLoader(dataset=testDataset, batch_size=batchSize, shuffle=False, num_workers=0, pin_memory=True)


# model
model = models.segmentation.deeplabv3_resnet50(num_classes=numClass, weight=None)
model.load_state_dict(torch.load("./DLCVHw1P2B_Epoch100.ckpt"))
model = model.to(device)


if __name__ == '__main__':
    model.eval()
    predictList = []

    with torch.no_grad():
        for imgs, fileNames in tqdm(testDataLoader):
            imgs = imgs.to(device)

            predict = model(imgs)['out'] 
            
            _, predicted = predict.max(1)
            predictList.extend(zip(fileNames, predicted.cpu().numpy()))
                     

    for fileNames, predict in predictList:
        predictImg = np.zeros((512, 512, 3), dtype=np.uint8)
        predictImg[np.where(predict == 0)] = cls_color[0]
        predictImg[np.where(predict == 1)] = cls_color[1]
        predictImg[np.where(predict == 2)] = cls_color[2]
        predictImg[np.where(predict == 3)] = cls_color[3]
        predictImg[np.where(predict == 4)] = cls_color[4]
        predictImg[np.where(predict == 5)] = cls_color[5]
        predictImg[np.where(predict == 6)] = cls_color[6]
        imageio.imwrite(os.path.join(output_images_directory, fileNames), predictImg)

        # img = Image.fromarray(np.uint8(predictImg))
        # img.save(os.path.join(output_images_directory, fileNames))