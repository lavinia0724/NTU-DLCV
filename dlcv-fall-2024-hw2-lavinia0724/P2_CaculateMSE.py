import os
import sys
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

def calculate_mse_loss(generated_dir, ground_truth_dir):
    # List of generated images
    generated_images = [f for f in os.listdir(generated_dir) if f.endswith(('.png'))]
    generated_images.sort()  # Ensure consistent ordering

    mse_losses = []

    for img_name in generated_images:
        gen_img_path = os.path.join(generated_dir, img_name)
        gt_img_path = os.path.join(ground_truth_dir, img_name)

        if not os.path.exists(gt_img_path):
            print(f"Ground truth image for {img_name} not found. Skipping.")
            continue

        # Load images and convert to RGB
        gen_img = Image.open(gen_img_path)
        gt_img = Image.open(gt_img_path)


        # Convert images to tensors
        gen_img_tensor = torch.from_numpy(np.array(gen_img)).float().permute(2, 0, 1) 
        gt_img_tensor = torch.from_numpy(np.array(gt_img)).float().permute(2, 0, 1)

        # Compute MSE loss
        mse_loss = F.mse_loss(gen_img_tensor, gt_img_tensor).item()
        mse_losses.append((img_name, mse_loss))

        print(f"MSE loss for {img_name}: {mse_loss:.6f}")

    # Calculate average MSE loss

    avg_mse = sum(loss for _, loss in mse_losses) / len(mse_losses)
    print(f"\nAverage MSE loss over {len(mse_losses)} images: {avg_mse:.6f}")


if __name__ == "__main__":
    # Paths to the directories containing generated images and ground truth images
    generated_images_dir = "./hw2/DDIM/output_images"
    ground_truth_dir = "./hw2_data/face/GT"    


    calculate_mse_loss(generated_images_dir, ground_truth_dir)
