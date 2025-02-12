import os
import json
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F


image_folder = "./hw3_data/p3_data/images"
# Generalized instruction for describing any type of image
instruction = (
    
    # "Provide a direct description of the main subject and what they are doing. Focus on what they are doing, do not mention their appearance, clothing, or colors. "
    # "Avoid stating unnecessary details like 'the main subject is'. Be concise and focus on the action."
    # "Provide a direct description of the main subject and what they are doing. "
    # "Avoid stating unnecessary details like 'the main subject is'. Be concise and focus on the action."
    # "Provide a direct description of the main subject and what they are doing. \n The caption example: A man in a baseball game running to base and others trying to tag him out."
    # "Provide a direct description of the main subject and what they are doing. \n The caption example: A cat curled up by the keyboard of a laptop."
    # "Write a single sentence describing the main action or content in this image. Focus on who/what is in the scene and what they are doing. Be concise and direct, similar to these examples:\n'A man in a baseball game running to base and others trying to tag him out.'\n'A box with half a dozen glazed and frosted donuts.'\n'A person at a table is eating a small pizza."
    "Write a single sentence that describing the main action or the main content in the image, focusing on who or what is in the scene and the action what they are doing. Examples:\n'A man in a baseball game running to base and others trying to tag him out.'\n'A box with half a dozen glazed and frosted donuts.'\n'A person at a table is eating a small pizza."
)
generation_config = {
    'max_new_tokens': 50,    # Moderate length to balance detail and conciseness  
    'do_sample': False,      # Enable sampling for diversity
    'temperature': 0.7,      # Set temperature to balance between randomness and focus
    'num_beams': 2,          # Increase number of beams for better candidate selection
    'eos_token_id': None,    # Will set below
    'pad_token_id': None,    # Will set below
}

# Load the processor to get the tokenizer special tokens

processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
generation_config['eos_token_id'] = processor.tokenizer.eos_token_id
generation_config['pad_token_id'] = processor.tokenizer.pad_token_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# captions_output = generate_captions(image_folder, instruction, generation_config)

# def generate_captions(image_folder, instruction, generation_config, output_file='captions_output.json', device='cuda'):
    # Load the pretrained LLaVA model for inference
model_id = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device)
model.eval()  # Ensure the model is in evaluation mode

# Load the processor
processor = AutoProcessor.from_pretrained(model_id)

# Initialize a dictionary to store the results
# captions_dict = {}

# Get a list of image files in the directory
# image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

# Use tqdm to display a progress bar
# for image_file in tqdm(image_files, desc='Processing images'):
#     image_path = os.path.join(image_folder, image_file)
#     filename = os.path.splitext(image_file)[0]  # Extract filename without extension

    # Prepare the conversation with the proper instruction
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": instruction},
            {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        # print(prompt)

def gen_attn(attn_layer=-1):
    raw_image = Image.open('./hw3_data/p3_data/images/umbrella.jpg').convert('RGB')
    inputs = processor(images=raw_image, text="USER: <image> Write a single sentence that describing the main action or the main content in the image, focusing on who or what is in the scene and the action what they are doing. Examples:\n'A man in a baseball game running to base and others trying to tag him out.'\n'A box with half a dozen glazed and frosted donuts.'\n'A person at a table is eating a small pizza. ASSISTANT:", return_tensors='pt').to(0)
    
    output = model.generate(**inputs, **generation_config, output_attentions=True, return_dict_in_generate=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    image_token_pos = (inputs['input_ids'][0] == image_token_id).nonzero()[0].item()

    img = inputs['pixel_values']
    img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))
    img = img[0].permute(1, 2, 0).detach().cpu().numpy()

    num_imges = len(output['attentions']) - 1
    num_cols = 5
    num_rows = (num_imges + num_cols - 1) // num_cols  # Limit rows
    fig = plt.figure(figsize=[10, 2 * num_rows])

    plt.subplot(num_rows, num_cols, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('<start>')

    last_decoded = ''

    for i in range(1, len(output['attentions'])):
        att = output['attentions'][i] 
        offset = len(output['attentions']) - i - 1
        decoded = processor.decode(output['sequences'][0][:(-offset or None)])
        if 'ASSISTANT:' in decoded:
            decoded = decoded.split('ASSISTANT:')[1]
        else:
            decoded = decoded.split('USER:')[1].split('<image>')[1]
        print(decoded)

        split_decoded = decoded[len(last_decoded):].strip()
        last_decoded = decoded

        att = att[attn_layer][0, :, 0, :]
        
        head_idx = 16  # Fixed head_idx
        num_heads = att.shape[0]
        
        # fig, ax = plt.subplots(figsize=(10, 10))
        
        image_att = att[:, image_token_pos:image_token_pos+576].reshape(-1, 24, 24)[head_idx]
        image_att = F.interpolate(image_att[None, None, ...], size=img.shape[:2], mode='bicubic')
        image_att = image_att.squeeze().detach().cpu().numpy()

        if i + 1 <= num_rows * num_cols:
            plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(img)
        plt.imshow(image_att, cmap='jet', alpha=0.5)
        plt.axis('off')
        plt.title(split_decoded)

    plt.tight_layout()
    plt.show()


# Example usage``
if __name__ == "__main__":
    gen_attn()