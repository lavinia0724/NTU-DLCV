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
# from byol_pytorch import BYOL
# import kornia.augmentation 
import sys
import csv

import warnings
warnings.filterwarnings("ignore")

# Data loader
class DLCVDataset(Dataset):
    def __init__(self, path, dataFolder, csvFile):
        self.path = path
        self.dataFolder = dataFolder
        
        self.DataFrame = pd.read_csv(csvFile)

        self.fileNames = self.DataFrame['filename'].tolist()
        self.ids = self.DataFrame['id'].tolist()
        self.labels = []  # For test data, labels can be an empty list

        self.Transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.fileNames)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.dataFolder, self.fileNames[index]))
        img = self.Transform(img)

        imgId = self.ids[index]
        fileName = self.fileNames[index]
        
        return img, imgId, fileName


path_of_the_images_csv_file = sys.argv[1]
path_of_the_folder_containing_images = sys.argv[2]
path_of_output_csv_file = sys.argv[3]


# hyper parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 100
batchSize = 256
learningRate = 1e-3


model = torchvision.models.resnet50(weights=None)
model.fc = torch.nn.Linear(2048, 65)
model.load_state_dict(torch.load("./DLCVHw1P1C_BestModel.ckpt"))  # 載入 pretrain model
model = model.to(device)


optimizer = torch.optim.Adam(model.parameters() , lr=learningRate, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()


# data
testDataset = DLCVDataset(path_of_the_folder_containing_images, path_of_the_folder_containing_images, path_of_the_images_csv_file)
testDataLoader = DataLoader(dataset=testDataset, batch_size=batchSize, shuffle=False, num_workers=0, pin_memory=True)


# Training
if __name__ == '__main__':
    # test --------------------------------------------------------------------
    model.eval()
    LabelList = []

    with torch.no_grad():
        for imgs, ids, fileNames in tqdm(testDataLoader):
            imgs = imgs.to(device)
            
            predict = model(imgs)

            _, predicted = predict.max(1)
            LabelList.extend(zip(ids.cpu().numpy(), fileNames, predicted.cpu().numpy()))


test_pred = pd.DataFrame(LabelList, columns=['id', 'filename', 'label'])
test_pred.to_csv(path_of_output_csv_file, index=False)

    