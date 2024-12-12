import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from torchvision.utils import save_image, make_grid

import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from UNet_conditional import UNet_conditional

import warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(600)  

output_folder = "./hw2_output/P1_Test/Visualization"

# Data loader
class DLCVDataset(Dataset):
    def __init__(self, mnistm_path, svhn_path, mode='train'):
        self.mode = mode

        # 讀取 MNIST-M Dataset
        self.mnistm_labelDf = pd.read_csv(os.path.join(mnistm_path, 'train.csv'))
        self.mnistm_fileNames = self.mnistm_labelDf['image_name'].tolist()
        self.mnistm_labels = self.mnistm_labelDf['label'].tolist()
        self.mnistm_files = [os.path.join(os.path.join(mnistm_path, 'data'), f) for f in self.mnistm_fileNames]
        self.mnistm_dataset_label = [0] * len(self.mnistm_files)  # "0" for MNIST-M

        # 讀取 SVHN Dataset
        self.svhn_labelDf = pd.read_csv(os.path.join(svhn_path, 'train.csv'))
        self.svhn_fileNames = self.svhn_labelDf['image_name'].tolist()
        self.svhn_labels = self.svhn_labelDf['label'].tolist()
        self.svhn_files = [os.path.join(os.path.join(svhn_path, 'data'), f) for f in self.svhn_fileNames]
        self.svhn_dataset_label = [1] * len(self.svhn_files)  # "1" for SVHN

        # 合併 MNIST-M 和 SVHN Dataset
        self.files = self.mnistm_files + self.svhn_files
        self.labels = self.mnistm_labels + self.svhn_labels
        self.dataset_labels = self.mnistm_dataset_label + self.svhn_dataset_label

        self.trainTransform = transforms.Compose([
            transforms.Resize((imageSize, imageSize)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        img = Image.open(file)

        img = self.trainTransform(img)
        
        return img, self.labels[index], self.dataset_labels[index]


# Define the DDPM Model with Classifier-Free Diffusion Guidance
class Diffusion:
    def __init__(self, model, noise_steps=1001, beta_start=1e-4, beta_end=0.02, img_size=28, num_classes=10, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.model = model.to(device)

        self.num_classes = num_classes
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, dataset_labels, cfg_scale=3, visualize_steps=None, initial_noise=None):
        model.eval()
        images = {}
        with torch.no_grad():
            x = initial_noise if initial_noise is not None else torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(0, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                
                # Conditional noise prediction
                predicted_noise_cond = model(x, t, labels, dataset_labels)
                
                # Unconditional noise prediction (for guidance)
                predicted_noise_uncond = model(x, t, None, None)

                # Apply guidance
                predicted_noise = torch.lerp(predicted_noise_uncond, predicted_noise_cond, cfg_scale)
                
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

                if visualize_steps and i in visualize_steps:
                    images[i] = x.clone()
        return x, images

# Hyper parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imageSize = 28
numClasses = 10

# Load the trained model
model = UNet_conditional(c_in=3, c_out=3, num_classes=numClasses).to(device)
model_path = "./hw2_output/P1_Train/DLCVHw2P1_ModelEpoch1000.pth"  # 載入訓練好的模型
model.load_state_dict(torch.load(model_path))
model.eval()

# Set up the diffusion process
ddpm = Diffusion(model=model, img_size=imageSize, device=device)

# Output folders
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

mnistm_output_folder = os.path.join(output_folder, "mnistm")
svhn_output_folder = os.path.join(output_folder, "svhn")

if not os.path.exists(mnistm_output_folder):
    os.makedirs(mnistm_output_folder)

if not os.path.exists(svhn_output_folder):
    os.makedirs(svhn_output_folder)

# Function to visualize reverse process
def visualize_reverse_process(output_folder, dataset_label, num_classes=10, samples_per_class=50, visualize_steps=[0, 200, 400, 600, 800, 1000]):
    for digit in range(num_classes):
        labels = torch.tensor([digit] * samples_per_class).long().to(device)
        dataset_labels = torch.tensor([dataset_label] * samples_per_class).long().to(device)  # 0 for MNIST-M, 1 for SVHN
        
        # Use the same initial noise for consistency
        initial_noise = torch.randn((samples_per_class, 3, imageSize, imageSize)).to(device)
        
        # Sample images and capture intermediate steps
        with torch.no_grad():
            _, sampled_images = ddpm.sample(model=model, n=len(labels), labels=labels, dataset_labels=dataset_labels, visualize_steps=visualize_steps, initial_noise=initial_noise)

        # Save images at different time steps for digit 0 only
        if digit == 0:
            for t, img in sampled_images.items():
                save_image(img[0], os.path.join(output_folder, f"0_t{t:04d}.png"))

# Visualize reverse process for MNIST-M
visualize_reverse_process(mnistm_output_folder, dataset_label=0)

# Visualize reverse process for SVHN
visualize_reverse_process(svhn_output_folder, dataset_label=1)
