import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import timm
import clip
from decoder import Decoder, Config
from tokenizer import BPETokenizer
from tqdm import tqdm
import sys

class CLIPScoreCalculator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

    def get_clip_score(self, image, caption):
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        text_input = clip.tokenize([caption], truncate=True).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_input)
        
        cos_sim = torch.nn.functional.cosine_similarity(image_features, text_features).item()
        return 2.5 * max(cos_sim, 0)

class DLCVDataset(Dataset):
    def __init__(self, imagesPathRoot, tokenizer, transform=None, max_length=50):
        self.imagesPathRoot = imagesPathRoot
        self.transform = transform
        self.images = os.listdir(self.imagesPathRoot)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        imgPath = os.path.join(self.imagesPathRoot, self.images[idx])
        fileName = os.path.splitext(self.images[idx])[0]
        
        img = Image.open(imgPath).convert('RGB')
        img_transformed = self.transform(img)
        
        return {
            'image_transformed': img_transformed,
            'file_name': fileName,
            'image_original': img
        }

def custom_collate(batch):
    images_transformed = torch.stack([item['image_transformed'] for item in batch])
    file_names = [item['file_name'] for item in batch]
    original_images = [item['image_original'] for item in batch]
    
    return images_transformed, file_names, original_images

def inference(model_path, validationDataLoader, decoder, vision_encoder, tokenizer, device, max_length):
    # Initialize CLIP calculator and load model
    clip_calculator = CLIPScoreCalculator()
    state_dict = torch.load(model_path, map_location=device)
    decoder.load_state_dict(state_dict, strict=False)
    decoder.eval()

    clip_scores = {}
    
    # Start inference
    progress_bar = tqdm(validationDataLoader, desc="Calculating CLIP Scores")
    with torch.no_grad():
        for images, fileNames, original_images in progress_bar:
            images = images.to(device)
            visual_features = vision_encoder.forward_features(images).to(device)

            # Prepare initial input for decoder
            start_token = torch.tensor([50256], dtype=torch.long, device=device).repeat(images.size(0), 1)
            current_output = start_token

            for _ in range(max_length):
                outputs, _ = decoder(current_output, visual_features)
                next_token_logits = outputs[:, -1, :]
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                current_output = torch.cat((current_output, next_tokens), dim=1)

                if torch.all(next_tokens == 50256):
                    break

            # Process each image
            for i in range(images.size(0)):
                output_tokens = current_output[i].tolist()
                caption = tokenizer.decode(output_tokens)
                if '<|endoftext|>' in caption:
                    caption = caption.split('<|endoftext|>')[1].strip()
                
                clip_score = clip_calculator.get_clip_score(original_images[i], caption)
                clip_scores[fileNames[i]] = clip_score

    # Find and print highest and lowest scoring images
    highest_score_img = max(clip_scores.items(), key=lambda x: x[1])
    lowest_score_img = min(clip_scores.items(), key=lambda x: x[1])
    
    print(f"\nHighest CLIP score: {highest_score_img[0]} (Score: {highest_score_img[1]:.4f})")
    print(f"Lowest CLIP score: {lowest_score_img[0]} (Score: {lowest_score_img[1]:.4f})")

if __name__ == "__main__":
    # Setup device and parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batchSize = 1
    max_length = 50

    # Setup models and data
    tokenizer = BPETokenizer("./encoder.json", "./vocab.bpe")
    data_config = timm.data.resolve_model_data_config(timm.create_model('vit_huge_patch14_clip_quickgelu_224').default_cfg)
    vit_transform = timm.data.create_transform(**data_config, is_training=False)
    
    test_images_path = sys.argv[1]
    validationDataset = DLCVDataset(test_images_path, tokenizer, transform=vit_transform, max_length=max_length)
    validationDataLoader = DataLoader(
        validationDataset, 
        batch_size=batchSize, 
        shuffle=False,
        collate_fn=custom_collate
    )

    # Setup models
    vision_encoder = timm.create_model('vit_huge_patch14_clip_quickgelu_224', pretrained=True)
    vision_encoder = vision_encoder.to(device)
    for param in vision_encoder.parameters():
        param.requires_grad = False

    decoder_weights_path = sys.argv[3]
    cfg = Config(checkpoint=decoder_weights_path)
    decoder = Decoder(cfg, vision_encoder)
    decoder = decoder.to(device)

    model_path = "P2_model_epoch7_best.pth"
    inference(model_path, validationDataLoader, decoder, vision_encoder, tokenizer, device, max_length)