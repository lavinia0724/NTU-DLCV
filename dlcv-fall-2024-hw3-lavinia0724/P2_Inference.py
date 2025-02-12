import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import timm
from decoder import Decoder, Config
from tokenizer import BPETokenizer
from tqdm import tqdm
import numpy as np
import loralib as lora
import sys


class DLCVDataset(Dataset):
    def __init__(self, imagesPathRoot, tokenizer, transform=None, max_length=50):
        self.imagesPathRoot = imagesPathRoot
        # self.tokenizer = tokenizer
        self.transform = transform
        # self.captionData = captionData
        # self.max_length = max_length
        self.images = os.listdir(self.imagesPathRoot)

        # self.images = {img['id']: img['file_name'] for img in self.captionData['images']}
        # self.annotations = self.captionData['annotations']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # annotation = self.annotations[idx]
        # imgId = annotation['image_id']
        imgPath = os.path.join(self.imagesPathRoot, self.images[idx])
        fileName = os.path.splitext(self.images[idx])[0]

        img = Image.open(imgPath).convert('RGB')
        img = self.transform(img)

        return img, fileName


# Transformations for training and validation using ViT's data configuration
data_config = timm.data.resolve_model_data_config(timm.create_model('vit_huge_patch14_clip_quickgelu_224').default_cfg)
vit_transform = timm.data.create_transform(**data_config, is_training=False)

# Transformations
validation_transform = vit_transform

# Hyper parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batchSize = 1
max_length = 50

# Data Preparation
tokenizer = BPETokenizer("./encoder.json", "./vocab.bpe")

# with open("./hw3_data/p2_data/val.json", 'r') as f:
#     validationCaptionData = json.load(f)

test_images_path = sys.argv[1]
# validationDataset = DLCVDataset("./hw3_data/p2_data/images/val", tokenizer, transform=validation_transform, max_length=max_length)
validationDataset = DLCVDataset(test_images_path, tokenizer, transform=validation_transform, max_length=max_length)
validationDataLoader = DataLoader(validationDataset, batch_size=batchSize, shuffle=False)

# Vision Encoder (ViT)
vision_encoder = timm.create_model('vit_huge_patch14_clip_quickgelu_224', pretrained=True)
vision_encoder = vision_encoder.to(device)
for param in vision_encoder.parameters():
    param.requires_grad = False

# Transformer Decoder
decoder_weights_path = sys.argv[3]
# cfg = Config(checkpoint="./hw3_data/p2_data/decoder_model.bin")
cfg = Config(checkpoint=decoder_weights_path)
decoder = Decoder(cfg, vision_encoder)
decoder = decoder.to(device)

lora.mark_only_lora_as_trainable(decoder)

for name, param in decoder.named_parameters():
    if 'visual_projection' in name or 'ln' in name:
        param.requires_grad = True

print("Trainable Params: ", sum(p.numel() for p in decoder.parameters() if p.requires_grad))

def inference(model_path):
    # Load trained model
    state_dict = torch.load(model_path, map_location=device)
    decoder.load_state_dict(state_dict, strict=False)
    decoder.eval()

    predictions = []

    # Start inference
    progress_bar = tqdm(validationDataLoader, desc="Generating Captions")
    with torch.no_grad():
        for images, fileNames in progress_bar:
            images = images.to(device)

            # Extract visual features
            visual_features = vision_encoder.forward_features(images).to(device)

            # Prepare initial input for decoder (start token)
            start_token = torch.tensor([50256], dtype=torch.long, device=device).repeat(images.size(0), 1)
            current_output = start_token

            for _ in range(max_length):
                # Predict next token
                outputs, _ = decoder(current_output, visual_features)
                next_token_logits = outputs[:, -1, :]
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                current_output = torch.cat((current_output, next_tokens), dim=1)

                # Stop if all sequences end with <|endoftext|>
                if torch.all(next_tokens == 50256):
                    break

            # Decode the generated captions
            for i in range(images.size(0)):
                output_tokens = current_output[i].tolist()
                caption = tokenizer.decode(output_tokens)
                # Extract the first sentence between <|endoftext|> tokens
                if '<|endoftext|>' in caption:
                    caption = caption.split('<|endoftext|>')[1].strip()
                predictions.append({fileNames[i].replace('.jpg', ''): caption})



    # Save predictions to JSON file
    # output_file = f'P2_predicted_caption.json'
    output_file = sys.argv[2]
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({k: v for d in predictions for k, v in d.items()}, f, indent=2)
    print(f"Saved predictions to {output_file}")


if __name__ == "__main__":
    model_path = "P2_model_epoch7_best.pth"  # Update the path if needed
    inference(model_path)
