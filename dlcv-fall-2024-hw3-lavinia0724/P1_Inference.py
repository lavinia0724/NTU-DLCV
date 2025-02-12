import os
import json
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from tqdm import tqdm
import sys

def generate_captions(image_folder, output_path, instruction, generation_config, device='cuda'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
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
    captions_dict = {}

    # Get a list of image files in the directory
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    # Use tqdm to display a progress bar
    for image_file in tqdm(image_files, desc='Processing images'):
        image_path = os.path.join(image_folder, image_file)
        filename = os.path.splitext(image_file)[0]  # Extract filename without extension

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
        # prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        # print(prompt)

        # Load the image
        raw_image = Image.open(image_path).convert('RGB')

        # Prepare inputs
        inputs = processor(images=raw_image, text="USER: <image> Write a single sentence that describing the main action or the main content in the image, focusing on who or what is in the scene and the action what they are doing. Examples:\n'A man in a baseball game running to base and others trying to tag him out.'\n'A box with half a dozen glazed and frosted donuts.'\n'A person at a table is eating a small pizza. ASSISTANT:", return_tensors='pt').to(device, torch.float16)

        # Generate the caption with the specified generation configuration
        with torch.no_grad():  # Ensure no gradients are calculated
            output = model.generate(**inputs, **generation_config)

        # Decode the output and clean it to remove unwanted prefixes
        caption = processor.decode(output[0][2:], skip_special_tokens=True)
        # Remove instruction prefix and assistant markers if present
        if "ASSISTANT:" in caption:
            caption = caption.split("ASSISTANT:", 1)[1].strip()

        # Append the result to the dictionary with the filename as the key
        captions_dict[filename] = caption

    # Save the results to a JSON file
    with open(output_path, 'w') as f:
        json.dump(captions_dict, f, indent=2, ensure_ascii=False)

    return captions_dict

# Example usage``
if __name__ == "__main__":
    image_folder = sys.argv[1]
    output_path = sys.argv[2]
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

    captions_output = generate_captions(image_folder, output_path, instruction, generation_config)