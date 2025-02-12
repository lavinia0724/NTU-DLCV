import os
import json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import timm
import math
import numpy as np
from decoder import Decoder, Config
from tokenizer import BPETokenizer
import loralib as lora

class DLCVDataset(Dataset):
    def __init__(self, imagesPathRoot, tokenizer, transform=None):
        self.imagesPathRoot = imagesPathRoot
        self.transform = transform
        self.images = os.listdir(self.imagesPathRoot)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        imgPath = os.path.join(self.imagesPathRoot, self.images[idx])
        fileName = os.path.splitext(self.images[idx])[0]
        image = Image.open(imgPath).convert('RGB')
        original_image = np.array(image)  # 保存原始圖像
        if self.transform:
            image = self.transform(image)
        return image, original_image, fileName

def visualize_attention(model_path, image_dir, save_dir='attention_maps'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup models and tokenizer
    data_config = timm.data.resolve_model_data_config(
        timm.create_model('vit_huge_patch14_clip_quickgelu_224').default_cfg
    )
    transform = timm.data.create_transform(**data_config, is_training=False)
    tokenizer = BPETokenizer("./encoder.json", "./vocab.bpe")
    
    # Initialize models
    vision_encoder = timm.create_model('vit_huge_patch14_clip_quickgelu_224', pretrained=True)
    vision_encoder = vision_encoder.to(device)
    vision_encoder.eval()
    for param in vision_encoder.parameters():
        param.requires_grad = False
    
    cfg = Config(checkpoint="./hw3_data/p2_data/decoder_model.bin")
    decoder = Decoder(cfg, vision_encoder)
    decoder = decoder.to(device)

    
    
    # Load model weights
    state_dict = torch.load(model_path, map_location=device)
    decoder.load_state_dict(state_dict, strict=False)
    decoder.eval()
    
    # Create dataset and dataloader
    dataset = DLCVDataset(image_dir, tokenizer, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    lora.mark_only_lora_as_trainable(decoder)

    for name, param in decoder.named_parameters():
        if 'visual_projection' in name or 'ln' in name:
            param.requires_grad = True

    print("Trainable Params: ", sum(p.numel() for p in decoder.parameters() if p.requires_grad))
        
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for img_tensor, original_img, image_name in dataloader:
            img_tensor = img_tensor.to(device)
            visual_features = vision_encoder.forward_features(img_tensor)

            img_tensor_display = img_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
            img_tensor_display = (img_tensor_display - img_tensor_display.min()) / (img_tensor_display.max() - img_tensor_display.min() + 1e-4)
            # print(img_tensor_display.shape[1])
            
            # Generate caption
            start_token = torch.full((len(img_tensor), 1), 50256, device=device)
            current_output = start_token
            generated_sequence = []
            attentions = []
            
            # Generate caption and collect attention weights
            for _ in range(50):  # max length
                outputs, attention = decoder(current_output, visual_features)
                next_token_logits = outputs[:, -1, :]
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                current_output = torch.cat([current_output, next_tokens], dim=1)

                generated_sequence.append(next_tokens.item())
                attentions.append(attention)
                
                if next_tokens.item() == 50256:  # End token
                    break
            
            # Decode caption
            for i in range(img_tensor.size(0)):
                output_tokens = current_output[i].tolist()
                caption = tokenizer.decode(output_tokens)

            # caption = tokenizer.decode(current_output.tolist())
            caption = caption.split('<|endoftext|>')[1].strip()
            words = caption.split()
            
            print(f"\nProcessing image: {image_name[0]}")
            print(f"Generated caption: {caption}")
            
            num_heads = attentions[0].shape[1]  # Should be 12
            
            # Process each attention head separately
            for head_idx in range(num_heads):
                # Calculate number of rows and columns for subplot grid
                num_words = len(words)
                num_cols = 4  # Fixed number of columns
                num_rows = math.ceil(num_words / num_cols)
                
                # Create figure for this head
                fig = plt.figure(figsize=(20, 5 * num_rows))
                plt.suptitle(f'Attention Head {head_idx + 1}', fontsize=16)
                
                # Get original image dimensions
                
                # Process each word
                for word_idx, word in enumerate(words):
                    # Create subplot
                    ax = plt.subplot(num_rows, num_cols, word_idx + 1)
                    
                    # image_att = attentions[-1][0]
                    image_att = attentions[word_idx][0, head_idx, 257+word_idx, 1:257]

                    # patch = int(math.sqrt(image_att.shape[0]))
                    image_att = image_att.reshape(16, 16)
                    
                    # Upscale attention map to original image size
                    image_att1 = F.interpolate(
                        torch.tensor(image_att, dtype=torch.float32)[None, None],
                        size=(224, 224),
                        mode='bilinear'
                    ).squeeze().detach().cpu().numpy()
                    
                    # Normalize attention weights
                    # image_att = image_att.cpu().numpy()
                    image_att1 = (image_att1 - image_att1.min()) / (image_att1.max() - image_att1.min())
                    
                    # Plot
                    # img_tensor_display = img_tensor[0].cpu().permute(1, 2, 0).numpy()
                    ax.imshow(img_tensor_display)
                    ax.imshow(image_att1, cmap='jet', alpha=0.5)
                    ax.axis('off')
                    ax.set_title(f'Word: "{word}"')
                
                plt.tight_layout()
                
                # Save the figure for this head
                save_path = os.path.join(save_dir, f'{image_name[0]}_head_{head_idx+1}.png')
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                plt.close()
                
                print(f"Saved attention map for head {head_idx + 1} to {save_path}")

if __name__ == "__main__":
    model_path = "P2_model_epoch7_best.pth"
    image_dir = "./hw3_data/p3_data/images"
    visualize_attention(model_path, image_dir)