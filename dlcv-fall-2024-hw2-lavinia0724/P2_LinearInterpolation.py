import os
import sys
import torch
import torchvision
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from PIL import Image
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

class Diffusion:
    def __init__(self, model, noise_steps=1000, img_size=28, device="cuda"):
        self.noise_steps = noise_steps

        self.beta = beta_scheduler().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.model = model.to(device)
        self.device = device

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
            x = noise.to(self.device)
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

# Interpolation functions
def slerp(val, low, high):
    """
    Spherical linear interpolation (SLERP)
    """
    omega = torch.acos(torch.clamp(torch.dot(low / torch.norm(low), high / torch.norm(high)), -1, 1))
    so = torch.sin(omega)
    if so == 0.0:
        return (1.0 - val) * low + val * high  # LERP in case of zero angle
    return torch.sin((1.0 - val) * omega) / so * low + torch.sin(val * omega) / so * high

def lerp(val, low, high):
    """
    Linear interpolation (LERP)
    """
    return (1.0 - val) * low + val * high

# Hyper parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imageSize = 256
model = UNet().to(device)
ddim = Diffusion(model=model, img_size=imageSize, device=device)
ddim.model.load_state_dict(torch.load("./hw2_data/face/UNet.pt"))
ddim.model.eval()

if __name__ == "__main__":
    noise_0 = torch.load("./hw2_data/face/noise/00.pt")
    noise_1 = torch.load("./hw2_data/face/noise/01.pt")

    alphas = np.linspace(0.0, 1.0, 11)

    # SLERP interpolation
    slerp_images = []
    for alpha in alphas:
        slerp_noise = slerp(alpha, noise_0.flatten(), noise_1.flatten()).view_as(noise_0)
        slerp_image = ddim.sample_ddim(slerp_noise, eta=0)
        slerp_images.append(slerp_image.squeeze(0))
    slerp_grid = torchvision.utils.make_grid(slerp_images, nrow=len(alphas), normalize=True)
    torchvision.utils.save_image(slerp_grid, "./hw2/DDIM/output_images/slerp_interpolation.png")

    # LERP interpolation
    lerp_images = []
    for alpha in alphas:
        lerp_noise = lerp(alpha, noise_0, noise_1)
        lerp_image = ddim.sample_ddim(lerp_noise, eta=0)
        lerp_images.append(lerp_image.squeeze(0))
    lerp_grid = torchvision.utils.make_grid(lerp_images, nrow=len(alphas), normalize=True)
    torchvision.utils.save_image(lerp_grid, "./hw2/DDIM/output_images/lerp_interpolation.png")