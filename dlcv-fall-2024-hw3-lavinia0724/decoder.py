import math
import collections
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import loralib as lora

class Config:
    def __init__(self, checkpoint=None):
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.vocab_size = 50257
        self.block_size = 1024
        self.checkpoint = checkpoint
        self.vtoken_size = 257

class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_attn = lora.Linear(cfg.n_embd, 3 * cfg.n_embd, r=56)
        self.c_proj = lora.Linear(cfg.n_embd, cfg.n_embd, r=56)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.vtoken_size = cfg.vtoken_size  # Store vtoken_size from config
        size = cfg.block_size
        mask = torch.tril(torch.ones(size, size))
        mask[:cfg.vtoken_size, :cfg.vtoken_size] = 1
        self.register_buffer('bias', mask.view(1, 1, size, size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        # print(x.shape)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C)), att

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.mlp = nn.Sequential(collections.OrderedDict([
            ('c_fc', lora.Linear(cfg.n_embd, 4 * cfg.n_embd, r=56)),
            ('act', nn.GELU(approximate='tanh')),
            ('c_proj', lora.Linear(4 * cfg.n_embd, cfg.n_embd, r=56))
        ]))

    def forward(self, x):
        attn_output, att = self.attn(self.ln_1(x))
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x, att

class Decoder(nn.Module):
    def __init__(self, cfg, visual_encoder):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.vtoken_size = cfg.vtoken_size  # Store vtoken_size from config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe = nn.Embedding(cfg.block_size, cfg.n_embd),
            h = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd)
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.visual_encoder = visual_encoder

        # Visual feature projection
        self.visual_projection = nn.Sequential(
            nn.Linear(1280, cfg.n_embd),  
            nn.ReLU(),                        
            nn.Linear(cfg.n_embd, cfg.n_embd)  
        )

        # Load checkpoint
        if self.cfg.checkpoint is not None:
            state_dict = torch.load(self.cfg.checkpoint)
            transposed = ['.c_attn.weight', '.c_fc.weight', '.c_proj.weight']
            for key, value in state_dict.items():
                if any(key.endswith(w) for w in transposed):
                    state_dict[key] = value.t()
            self.transformer.load_state_dict(state_dict, strict=False)

    def forward(self, captions: Tensor, image_features: Tensor):
        # Process captions
        captions = captions.to(dtype=torch.long)
        captions = torch.narrow(captions, 1, 0, min(captions.size(1), self.block_size))

        token_embeddings = self.transformer.wte(captions)

        # Extract CLS token from image_features
        # image_features = image_features[:, 0, :]  # [batch_size, 768]

        # Project and reshape visual features
        visual_proj = self.visual_projection(image_features)  # [batch_size, n_embd]
        # visual_proj = visual_proj.unsqueeze(1)  # [batch_size, 1, n_embd]

        # Concatenate visual features with text embeddings
        visual_text = torch.cat([visual_proj, token_embeddings], dim=1)

        # Generate position embeddings
        pos = torch.arange(0, visual_text.size(1), dtype=torch.long, device=captions.device).unsqueeze(0)

        # Get token embeddings
        position_embeddings = self.transformer.wpe(pos)
        x = visual_text + position_embeddings

        visual_len = visual_proj.size(1)



        att_weights = []
        # Pass through transformer blocks
        for block in self.transformer.h:
            x, att = block(x)
            # att_weights.append(att)


        x = x[:, visual_len:]
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # print("forward att shape: ", att.shape)
        # att = att[:, :, self.vtoken_size:, 1:self.vtoken_size]
        # print("forward att shape after vtoken: ", att.shape)

        return logits, att