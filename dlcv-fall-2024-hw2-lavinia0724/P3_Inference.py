import argparse, os, sys, glob
import json
import cv2
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

input_json_path = sys.argv[1]
output_folder = sys.argv[2]
pretrained_model_path = sys.argv[3]


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

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

# def put_watermark(img, wm_encoder=None):
#     if wm_encoder is not None:
#         img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#         img = wm_encoder.encode(img, 'dwtDct')
#         img = Image.fromarray(img[:, :, ::-1])
#     return img

def main():
    # Replace argparse with direct variable assignment from sys.argv
    config_path = "./stable-diffusion/configs/stable-diffusion/v1-inference.yaml"
    ddim_steps = 50
    H = 512
    W = 512
    C = 4
    f = 8
    n_samples = 25
    scale = 7.5
    seed = 12818
    precision = "autocast"

    seed_everything(seed)

    config = OmegaConf.load(config_path)
    model = load_model_from_config(config, pretrained_model_path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    sampler = DPMSolverSampler(model)

    # Load input JSON
    with open(input_json_path, 'r') as input_file:
        input_data = json.load(input_file)

    # Begin of code segment to remain unchanged
    # Use data from index 0
    data_index = "0"
    input_prompts_0 = input_data[data_index]['prompt']
    token_name_0 = input_data[data_index]['token_name']

    # model.cond_stage_model.tokenizer.add_tokens(token_name_0)
    # num_embeddings = model.cond_stage_model.transformer.text_model.embeddings.token_embedding.num_embeddings
    # embedding_dim = model.cond_stage_model.transformer.text_model.embeddings.token_embedding.embedding_dim

    # embeddings_backup = model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight.clone().detach()
    # model.cond_stage_model.transformer.text_model.embeddings.token_embedding = nn.Embedding(num_embeddings+1, embedding_dim).to(device)
    # model.cond_stage_model.transformer.text_model.embeddings.token_embedding.requires_grad_(False)
    # model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight[:-1] = embeddings_backup

    # Use data from index 1
    data_index = "1"
    input_prompts_1 = input_data[data_index]['prompt']
    token_name_1 = input_data[data_index]['token_name']

    model.cond_stage_model.tokenizer.add_tokens(token_name_0)
    model.cond_stage_model.tokenizer.add_tokens(token_name_1)
    num_embeddings = model.cond_stage_model.transformer.text_model.embeddings.token_embedding.num_embeddings
    embedding_dim = model.cond_stage_model.transformer.text_model.embeddings.token_embedding.embedding_dim

    embeddings_backup = model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight.clone().detach()
    model.cond_stage_model.transformer.text_model.embeddings.token_embedding = nn.Embedding(num_embeddings+2, embedding_dim).to(device)
    model.cond_stage_model.transformer.text_model.embeddings.token_embedding.requires_grad_(False)
    model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight[:-2] = embeddings_backup

    # Load the trained embedding for the new token
    dog_embedding = torch.load("./P3_models/dog_epoch85.pt")
    model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight[-2] = dog_embedding

    David_embedding = torch.load("./P3_models/DavidRevoy_epoch110.pt")
    model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight[-1] = David_embedding
    # End of code segment to remain unchanged

    os.makedirs(output_folder, exist_ok=True)
    outpath = output_folder

    # print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    # wm = "StableDiffusionV1"
    # wm_encoder = WatermarkEncoder()
    # wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    precision_scope = autocast if precision == "autocast" else nullcontext

    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for data_index in ["0", "1"]: # change
                    if data_index == "0":
                        input_prompts = input_prompts_0
                        token_name = token_name_0
                    elif data_index == "1":
                        input_prompts = input_prompts_1
                        token_name = token_name_1

                    # Create source folder for current data index
                    source_folder = os.path.join(outpath, data_index)
                    os.makedirs(source_folder, exist_ok=True)

                    torch.manual_seed(seed)  # Add this line to set the seed for each prompt

                    for prompt_idx, prompt in enumerate(input_prompts):
                        prompt_folder = os.path.join(source_folder, str(prompt_idx))
                        os.makedirs(prompt_folder, exist_ok=True)

                        # Set seed before generating each prompt

                        for sample_idx in trange(n_samples, desc=f"Data index {data_index}, prompt {prompt_idx}"):
                            # if prompt_idx == 0:
                            #     continue

                            uc = None
                            if scale != 1.0:
                                uc = model.get_learned_conditioning(1 * [""])
                            c = model.get_learned_conditioning([prompt])
                            shape = [C, H // f, W // f]
                            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                             conditioning=c,
                                                             batch_size=1,
                                                             shape=shape,
                                                             verbose=False,
                                                             unconditional_guidance_scale=scale,
                                                             unconditional_conditioning=uc)

                            x_samples_ddim = model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                            x_checked_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)

                            # Save individual sample
                            x_sample = 255. * rearrange(x_checked_image_torch[0].cpu().numpy(), 'c h w -> h w c')
                            img = Image.fromarray(x_sample.astype(np.uint8))
                            # img = put_watermark(img, wm_encoder)
                            img.save(os.path.join(prompt_folder, f"source{data_index}_prompt{prompt_idx}_{sample_idx}.png"))

    print(f"Your samples are ready and waiting for you here: \n{outpath} \nEnjoy.")

if __name__ == "__main__":
    main()