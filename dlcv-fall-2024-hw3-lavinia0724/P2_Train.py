import os
import json
import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import timm
from decoder import Decoder, Config
from tokenizer import BPETokenizer
from tqdm import tqdm
import loralib as lora
import numpy as np

# Dataset class remains the same
class DLCVDataset(Dataset):
    def __init__(self, imagesPathRoot, tokenizer, captionData, transform=None, max_length=50):
        self.imagesPathRoot = imagesPathRoot
        self.tokenizer = tokenizer
        self.transform = transform
        self.captionData = captionData
        self.max_length = max_length

        self.images = {img['id']: img['file_name'] for img in self.captionData['images']}
        self.annotations = self.captionData['annotations']

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        imgId = annotation['image_id']
        fileName = self.images[imgId]
        imgPath = os.path.join(self.imagesPathRoot, fileName)

        img = Image.open(imgPath).convert('RGB')
        img = self.transform(img)        

        caption = annotation['caption']
        tokenized_caption = self.tokenizer.encode(caption, allowed_special=['<|endoftext|>'])
        padded_caption = [50256] + tokenized_caption
        
        if len(padded_caption) > self.max_length:
            padded_caption = padded_caption[:self.max_length]
        
        if len(padded_caption) < self.max_length:
            padded_caption = padded_caption + [50256] * (self.max_length - len(padded_caption))

        return img, torch.tensor(padded_caption, dtype=torch.long), fileName

# Data transformations and configuration
data_config = timm.data.resolve_model_data_config(timm.create_model('vit_huge_patch14_clip_quickgelu_224').default_cfg)
vit_transform = timm.data.create_transform(**data_config, is_training=False)

def train_transform(img):
    img = transforms.TrivialAugmentWide()(img)
    img = vit_transform(img)
    return img

validation_transform = vit_transform

# Training function
def train():
    # Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batchSize = 16
    base_lr = 4e-3
    max_lr = 4e-3
    epochs = 7
    max_length = 50

    # Initialize tokenizer and load data
    tokenizer = BPETokenizer("./encoder.json", "./vocab.bpe")
    
    with open("./hw3_data/p2_data/train.json", 'r') as f:
        trainCaptionData = json.load(f)

    trainDataset = DLCVDataset("./hw3_data/p2_data/images/train", tokenizer, trainCaptionData, 
                              transform=train_transform, max_length=max_length)
    trainDataLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)

    with open("./hw3_data/p2_data/val.json", 'r') as f:
        validationCaptionData = json.load(f)

    validationDataset = DLCVDataset("./hw3_data/p2_data/images/val", tokenizer, validationCaptionData, 
                                   transform=validation_transform, max_length=max_length)
    validationDataLoader = DataLoader(validationDataset, batch_size=batchSize, shuffle=False)

    # Initialize models
    vision_encoder = timm.create_model('vit_huge_patch14_clip_quickgelu_224', pretrained=True)
    vision_encoder = vision_encoder.to(device)
    for param in vision_encoder.parameters():
        param.requires_grad = False

    cfg = Config(checkpoint="./hw3_data/p2_data/decoder_model.bin")
    decoder = Decoder(cfg, vision_encoder)
    decoder = decoder.to(device)

    # Setup LoRA
    lora.mark_only_lora_as_trainable(decoder)
    for name, param in decoder.named_parameters():
        if 'visual_projection' in name or 'ln' in name:
            param.requires_grad = True

    trainable_params = [param for param in decoder.parameters() if param.requires_grad]
    print("Trainable Params: ", sum(p.numel() for p in trainable_params))

    # Loss and optimizer setup
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
    optimizer = optim.AdamW(trainable_params, lr=base_lr, weight_decay=1e-2)
    total_steps = epochs * len(trainDataLoader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, total_steps=total_steps,
        pct_start=0.7, anneal_strategy='cos'
    )

    # Training loop
    for epoch in range(epochs):
        decoder.train()
        train_loss = 0.0
        progress_bar = tqdm(trainDataLoader, desc=f"Epoch {epoch + 1}/{epochs}")
        
        for images, captions, _ in progress_bar:
            images, captions = images.to(device), captions.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                visual_features = vision_encoder.forward_features(images).to(device)
            
            # Get logits and attention weights from decoder
            logits, _ = decoder(captions, visual_features)
            logits = logits.reshape(-1, logits.size(-1))
            
            # Process captions for loss calculation
            processed_captions = []
            original_len = captions.size(1)
            for caption in captions:
                caption = caption[1:]
                caption = caption.to(device)

                idx = len(caption) - 1
                while idx > 0 and caption[idx] == 50256:
                    idx -= 1
                caption = caption[:idx + 2]

                pad_len = original_len - len(caption)
                if pad_len > 0:
                    pad_tensor = torch.full((pad_len,), -100, dtype=torch.long, device=device)
                    caption = torch.cat([caption, pad_tensor], dim=0)

                processed_captions.append(caption)

            processed_captions = torch.stack(processed_captions, dim=0)
            gt_tokens = processed_captions.view(-1)
            
            # Calculate loss and update
            loss = criterion(logits, gt_tokens)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 0.5)
            optimizer.step()
            scheduler.step()
            
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{current_lr:.2e}'})
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(trainDataLoader)
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Training Loss: {avg_train_loss:.4f}")
        
        # Validation phase
        decoder.eval()
        validation_loss = 0.0
        progress_bar = tqdm(validationDataLoader, desc="Validation")

        with torch.no_grad():
            for images, captions, _ in progress_bar:
                images, captions = images.to(device), captions.to(device)
                visual_features = vision_encoder.forward_features(images).to(device)
                logits, _ = decoder(captions, visual_features)
                logits = logits.reshape(-1, logits.size(-1))

                processed_captions = []
                original_len = captions.size(1)
                for caption in captions:
                    caption = caption[1:]
                    caption = caption.to(device)

                    idx = len(caption) - 1
                    while idx > 0 and caption[idx] == 50256:
                        idx -= 1
                    caption = caption[:idx + 2]

                    pad_len = original_len - len(caption)
                    if pad_len > 0:
                        pad_tensor = torch.full((pad_len,), -100, dtype=torch.long, device=device)
                        caption = torch.cat([caption, pad_tensor], dim=0)

                    processed_captions.append(caption)

                processed_captions = torch.stack(processed_captions, dim=0)
                gt_tokens = processed_captions.view(-1)
                loss = criterion(logits, gt_tokens)
                validation_loss += loss.item()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_validation_loss = validation_loss / len(validationDataLoader)
        print(f"Validation Loss: {avg_validation_loss:.4f}")

        # Save model
        trainable_weights = [name for name, param in decoder.named_parameters() if param.requires_grad]
        save_weights = {k: v for k, v in decoder.state_dict().items() if k in trainable_weights}
        model_file = f'P2_model_epoch{epoch+1}.pth'
        torch.save(save_weights, model_file)
        print(f"Saved model to {model_file}")

if __name__ == "__main__":
    train()