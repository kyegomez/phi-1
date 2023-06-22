import torch
import torch.nn as nn
from torchscale.torchscale import Decoder, DecoderConfig

import bitsandbytes

from torchscale.torchscale.component.embedding import PositionalEmbedding

import torch
from x_transformers.x_transformers import TransformerWrapper, Decoder, AutoregressiveWrapper



class phi1(nn.Module):
    def __init__(self):
        super().__init__()
        # Instantiate Clip Vit-l/14
        self.embed = bitsandbytes.nn.modules.Embedding(
            32002,
            2048,
            padding_idx=1
        )
        self.embed_positions= PositionalEmbedding(
            2048,
            2048,
            1
        )

        self.output_projection = torch.nn.Linear(
            2048, 32002, bias=False
        )
        torch.nn.init.normal_(
            self.output_projection.weight, mean=0, std=2048**-0.5
        )

        # Config following KOSMOS-1 paper (https://arxiv.org/pdf/2302.14045.pdf)
        self.config = DecoderConfig(
            decoder_layers=24,
            decoder_embed_dim=2048,
            decoder_ffn_embed_dim=8192,
            decoder_attention_heads=32,
            dropout=0.1,
            activation_fn="gelu",
            attention_dropout=0.1,
            vocab_size=64007,
            xpos_rel_pos=True,
            multiway=True,
            max_rel_pos=2048,

        )
        self.decoder = Decoder(
            self.config,
            embed_tokens=self.embed,
            embed_positions=self.embed_positions,
            output_projection=self.output_projection
        )

    def forward(self, text_tokens, images, **kwargs):
        images = self.clip_model(pixel_values=images)["last_hidden_state"]
        images = self.perceive(images).squeeze(1)
        images = self.image_proj(images)

        model_input = self.decoder.forward_embedding(text_tokens)[1]
        model_input = torch.cat([model_input[:, 0:2], images, model_input[:, 2:]], dim=1)
        model_input = self.decoder.forward_embedding(model_input, token_embedding=model_input)[0]

        return self.decoder(model_input, passed_x=model_input)[0]
    

#verison 1 using x transformers
phi2 = TransformerWrapper(
        num_tokens=64007,
        max_seq_len=8192,
        use_abs_pos_emb=False,
        # tokenizer=tokenizer, # !
        attn_layers = Decoder(
            dim=2048, # 2048
            depth=16, # 16
            dim_head=128,
            heads=8,
            rotary_xpos=True,
            attn_flash = True,
            qk_norm=True,
            attn_qk_norm=True,
            attn_qk_norm_dim_scale=True # set this to True, in addition to `attn_qk_norm = True`
        )
    )

phi2 = AutoregressiveWrapper(phi2)