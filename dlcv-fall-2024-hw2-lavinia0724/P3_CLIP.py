import os
import json
import clip
import torch
from PIL import Image

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Load id-to-label mapping
with open('./hw2_data/clip_zeroshot/id2label.json', 'r') as f:
    id2label = json.load(f)

# Define the image folder path
image_folder = "./hw2_data/clip_zeroshot/val"

# Prepare the text inputs
text_inputs = torch.cat([clip.tokenize(f"A photo of {label}") for label in id2label.values()]).to(device)

# Iterate through images and classify
results = {}
correct_predictions = 0
total_images = 0

for image_name in os.listdir(image_folder):
    if image_name.endswith(".png"):
        # Extract class_id and image_id from the file name
        class_id, image_id = image_name.split("_")
        image_path = os.path.join(image_folder, image_name)

        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)

        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

        # Normalize the features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Calculate similarity and get the most similar label
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(1)

        # Get predicted label
        predicted_label = list(id2label.values())[indices[0].item()]
        results[image_name] = predicted_label

        # Check if the prediction is correct
        if predicted_label == id2label[class_id]:
            correct_predictions += 1
        total_images += 1

        # Print the result
        # print(f"Image: {image_name}, Predicted label: {predicted_label}, Confidence: {values[0].item():.2f}%")

# Calculate and print accuracy
accuracy = correct_predictions / total_images * 100
print(f"Accuracy: {accuracy:.2f}%")

# Optionally, save the results to a file
with open("classification_results.json", "w") as f:
    json.dump(results, f, indent=4)
