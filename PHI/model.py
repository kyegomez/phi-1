from Phi.core.transformer import Transformer, Decoder
from Phi.core.autoregressive_wrapper import AutoregressiveWrapper 

Phi = Transformer(
    num_tokens=64007,
    max_seq_len=8192,
    use_abs_pos_emb=False,
    attn_layers = Decoder(
        dim=2560, # 2048
        depth=32, # 16
        dim_head=128,
        heads=24,
        alibi_pos_bias=True,
        alibi_num_heads=12,
        rotary_xpos=True,
        attn_flash = True,
        attn_one_kv_head = True,
        qk_norm=True,
        attn_qk_norm=True,
        attn_qk_norm_dim_scale=True # set this to True, in addition to `attn_qk_norm = True`
    )
)

Phi = AutoregressiveWrapper(Phi)