import argparse, os, glob
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
from torch.optim import Adam
import torch.nn.functional as F
from ldm.util import instantiate_from_config
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torchvision import transforms

# Function to load the model from config
def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

# Main function
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="./stable-diffusion/configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="./stable-diffusion/ldm/models/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./outputs",
        help="dir to write results to",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    opt = parser.parse_args()

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    sampler = DPMSolverSampler(model)

    # Add new token to the tokenizer
    model.cond_stage_model.tokenizer.add_tokens('<new1>')
    new_token_id = model.cond_stage_model.tokenizer.convert_tokens_to_ids('<new1>')

    # Resize the token embeddings to accommodate the new token
    embedding_layer = model.cond_stage_model.transformer.text_model.embeddings.token_embedding
    old_num_tokens, embedding_dim = embedding_layer.weight.shape

    with torch.no_grad():
        # Create new embedding weights with an extra row for the new token
        new_embedding_weight = torch.cat(
            [embedding_layer.weight.data, torch.randn(1, embedding_dim).to(device) * 0.01],
            dim=0
        )
    # Assign the new weights to the embedding layer
    embedding_layer.weight = nn.Parameter(new_embedding_weight)
    embedding_layer.num_embeddings = old_num_tokens + 1  # Update the num_embeddings

    # Ensure the entire embedding layer requires gradients
    embedding_layer.weight.requires_grad = True

    # Define the optimizer for the embedding layer
    optimizer = Adam([embedding_layer.weight], lr=1e-4)

    # Prompts to be used for training
    prompts = [
        "A joyful portrait of <new1> with its tongue out",
        "A cute and fluffy <new1> enjoying a bright day",
        "A photo of <new1> smiling happily under blooming flowers",
        "A charming <new1> posing with its tongue hanging out",
        "A happy <new1> with fluffy fur and a bright expression",
        "A photo capturing a playful <new1> on a sunny day",
        "A lovely picture of <new1> with vibrant background colors",
        "An adorable <new1> looking relaxed and joyful",
        "A fluffy <new1> in front of a blurred floral backdrop",
        "A cute <new1> sitting gracefully with a bright smile"
    ]

    # Load dog images for training
    image_paths = glob.glob(r"./hw2_data/textual_inversion/0/*.jpg")
    assert len(image_paths) == 5, "There should be exactly 5 dog images."

    # Augmentation to enhance generalization of learned features
    augmentation = T.Compose([
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.RandomRotation(degrees=15),
        T.ToTensor()
    ])

    # Training loop
    mse_loss = nn.MSELoss()
    n_epochs = 500  # Number of training epochs
    for epoch in range(n_epochs):
        total_loss = 0
        count = 0
        for prompt in tqdm(prompts, desc=f'Epoch {epoch + 1}/{n_epochs}'):
            for image_path in image_paths:
                # Load the image and convert it to tensor
                image = Image.open(image_path).convert("RGB")
                image = image.resize((opt.W, opt.H))
                image = augmentation(image).unsqueeze(0).to(device)

                # Encode the image to latent space
                with torch.no_grad():
                    latent = model.encode_first_stage(image)
                    latent = model.get_first_stage_encoding(latent)

                # Generate a noisy latent sample
                timesteps = torch.randint(0, model.num_timesteps, (1,), device=device).long()
                noise = torch.randn_like(latent).to(device)
                noisy_latent = model.q_sample(latent, timesteps, noise=noise)

                # Get the conditioning for the prompt
                c = model.get_learned_conditioning([prompt])

                # Predict the noise from the model
                predicted_noise = model.apply_model(noisy_latent, timesteps, c)

                # Calculate the loss using Mean Squared Error
                loss = mse_loss(predicted_noise, noise)

                # Backpropagate the loss to update the new token embedding
                optimizer.zero_grad()
                loss.backward()

                # Zero out gradients for all embeddings except the new one
                with torch.no_grad():
                    embedding_layer.weight.grad[:-1] = 0  # Keep gradient for the last embedding only

                optimizer.step()

                total_loss += loss.item()
                count += 1

                # Optional: Verify that gradients are being updated
                # Uncomment the following lines to print gradient norms
                # grad_norm = embedding_layer.weight.grad[-1].norm().item()
                # print(f"Gradient norm for new embedding: {grad_norm}")

        # Print the average loss at every epoch
        avg_loss = total_loss / count if count > 0 else 0
        print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {avg_loss:.4f}")

        # Save the embedding every 10 epochs
        if (epoch + 1) % 10 == 0:
            epoch_embedding_path = os.path.join(opt.outdir, f"dog_epoch{epoch + 1}.pt")
            torch.save(embedding_layer.weight[-1].detach().cpu(), epoch_embedding_path)
            print(f"Intermediate embedding saved to {epoch_embedding_path}")

    # Save the trained embedding
    embedding_path = os.path.join(opt.outdir, "dog.pt")
    torch.save(embedding_layer.weight[-1].detach().cpu(), embedding_path)
    print(f"Trained embedding saved to {embedding_path}")

if __name__ == "__main__":
    main()



