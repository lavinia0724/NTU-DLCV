import os
import sys
import torch
# import torch.nn as nn
# import torch.optim as optim
import torchvision 
# from torch.utils.data import DataLoader, Dataset

# from torchvision.utils import save_image, make_grid

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import random
from UNet import UNet
from utils import beta_scheduler

import warnings
warnings.filterwarnings("ignore")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(600)  


path_directory_predefined_noises = sys.argv[1]
path_directory_generated_images = sys.argv[2]
path_pretrained_model_weight = sys.argv[3]


class Diffusion:
    def __init__(self, model, noise_steps=1000, img_size=32, device="cuda"):
        self.noise_steps = noise_steps

        self.beta = beta_scheduler().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.model = model.to(device)
        self.device = device

    # def prepare_noise_schedule(self):
    #     return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    # Reference: https://github.com/ermongroup/ddim/blob/main/functions/denoising.py
    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    def compute_alpha(self, beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def sample_ddim(self, noise, time_steps=50, eta=0.0):
        model = self.model

        # Redefine seq using from 1000 to 1, timesteps = 50
        interval = self.noise_steps // time_steps
        seq = [int(s) for s in list(range(0, self.noise_steps, interval))]
        seq_next = [-1] + list(seq[:-1])

        model.eval()

        with torch.no_grad():
            x = torch.load(noise).to(self.device)
            n = x.size(0)
            x0_preds = []
            xs = [x]
            
            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                
                # Compute alpha values at timestep t and next_t
                at = self.compute_alpha(self.beta, t.long())
                at_next = self.compute_alpha(self.beta, next_t.long())
                
                xt = xs[-1].to(self.device)  # Last generated image
                et = model(xt, t)  # Predict noise
                
                # Compute x_0 prediction
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                x0_preds.append(x0_t.to('cpu'))  # Store x_0 predictions
                
                # Compute the coefficients for the next step
                c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                
                # Generate the next step image xt_next
                xt_next = (at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et).float()
                xs.append(xt_next.to('cpu'))  # Store next xt
            
        return xs[-1]

    


# Hyper parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# batchSize = 64
imageSize = 32
# numClasses = 10


model = UNet().to(device)
ddim = Diffusion(model=model, img_size=imageSize, device=device)
ddim.model.load_state_dict(torch.load(path_pretrained_model_weight))
ddim.model.eval()


# outputPath = "./hw2_output/P1_Train"
if not os.path.exists(path_directory_generated_images):
        os.makedirs(path_directory_generated_images)



if __name__ == "__main__":
    noises = os.listdir(path_directory_predefined_noises)
                    
    for noise in noises:
        x = ddim.sample_ddim(os.path.join(path_directory_predefined_noises, noise), eta=0)
        fileName = os.path.splitext(os.path.basename(noise))[0]
        torchvision.utils.save_image(x, os.path.join(path_directory_generated_images, f"{fileName}.png"), normalize=True)
    
