import os
import sys
import torch
import torchvision
import numpy as np
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

# Paths from command-line arguments
path_directory_predefined_noises = sys.argv[1]
path_directory_generated_images = sys.argv[2]
path_pretrained_model_weight = sys.argv[3]

class Diffusion:
    def __init__(self, model, noise_steps=1000, img_size=28, device="cuda"):
        self.noise_steps = noise_steps
        self.beta = beta_scheduler().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.img_size = img_size
        self.model = model.to(device)
        self.device = device

    def compute_alpha(self, beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def sample_ddim(self, noise, time_steps=50, eta=0.0):
        model = self.model
        model.eval()

        # Redefine seq using numpy.linspace
        interval = self.noise_steps // time_steps
        seq = [int(s) for s in list(range(0, self.noise_steps, interval))]
        seq_next = [-1] + list(seq[:-1])

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

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imageSize = 256

# Initialize model and diffusion
model = UNet().to(device)
ddim = Diffusion(model=model, img_size=imageSize, device=device)
ddim.model.load_state_dict(torch.load(path_pretrained_model_weight, map_location=device))
ddim.model.eval()

# Create output directory if it doesn't exist
if not os.path.exists(path_directory_generated_images):
    os.makedirs(path_directory_generated_images)

if __name__ == "__main__":
    # Define eta values and noise files
    eta_list = [0.0, 0.25, 0.5, 0.75, 1.0]
    noise_files = ['00.pt', '01.pt', '02.pt', '03.pt']
    noise_paths = [os.path.join(path_directory_predefined_noises, noise) for noise in noise_files]

    # Initialize a list to store all generated images
    all_images = []

    for eta in eta_list:
        images_per_eta = []
        for noise_path in noise_paths:
            # Load noise tensor
            noise = torch.load(noise_path).to(device)
            # Generate image using DDIM sampler with the specified eta
            generated_image = ddim.sample_ddim(noise, eta=eta)
            # Normalize the image to [0,1] range
            generated_image = (generated_image.clamp(-1, 1) + 1) / 2  # Assuming the model outputs images in [-1,1]

            # Ensure the generated image has 3 dimensions [C, H, W]
            if generated_image.dim() == 4:
                # Remove the batch dimension if present
                generated_image = generated_image.squeeze(0)

            # Check if the channel dimension is in the correct position
            if generated_image.shape[0] != 3:
                # If the tensor shape is [H, W, C], permute it to [C, H, W]
                generated_image = generated_image.permute(2, 0, 1)

            # Verify the shape
            print(f"Adjusted image shape for eta={eta}, noise={os.path.basename(noise_path)}: {generated_image.shape}")

            images_per_eta.append(generated_image)
        # Stack images for the current eta
        images_per_eta = torch.stack(images_per_eta)
        all_images.append(images_per_eta)

    # Concatenate all images
    all_images = torch.cat(all_images, dim=0)
    print(f"Shape of all_images before make_grid: {all_images.shape}")

    # Make a grid image
    grid = torchvision.utils.make_grid(all_images, nrow=len(noise_files))

    # Save the grid image
    grid_image_path = os.path.join(path_directory_generated_images, 'combined_grid.png')
    torchvision.utils.save_image(grid, grid_image_path)

    print(f"Combined grid image saved to {grid_image_path}")