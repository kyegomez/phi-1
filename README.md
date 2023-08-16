[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

Since Phi is ready to train Agora is actively seeking cloud providers or grant providers to train this all-new revolutionary model and release it open source, if you would like to learn more please email me at `kye@apac.ai`

# Phi: Ultra-Fast and Ultra-Intelligent SOTA Language Model 🚀🌌

[Textbooks Are All You Need](https://arxiv.org/abs/2306.11644)

Phi is a state-of-the-art language model that pushes the boundaries of natural language understanding and generation. Designed for high performance and efficiency, Phi is built upon advanced techniques that make it a strong contender against the likes of OpenAI's GPT-4 and PALM.



# Usage
Get started:

1. Clone the repository and install the required packages.


```
git clone https://github.com/kyegomez/Phi
cd Phi
pip3 install -r requirements.txt
cd Phi
python3 training_distributed.py
```

# Training

First:

`Accelerate Config`

Enable Deepspeed 3: 

`Accelerate launch train_distributed_accelerate.py`



## Dataset building building

Data
You can preprocess a different dataset in a way similar to the C4 dataset used during training by running the build_dataset.py script. This will pre-tokenize, chunk the data in blocks of a specified sequence length, and upload to the Huggingface hub. For example:

```python3 Phi/build_dataset.py --seed 42 --seq_len 8192 --hf_account "HUGGINGFACE APIKEY" --tokenizer "EleutherAI/gpt-neox-20b" --dataset_name "EleutherAI/the_pile_deduplicated"```



# Inference

```python3 inference.py "My dog is very cute" --seq_len 256 --temperature 0.8 --filter_thres 0.9 --model "phi"``` 

Not yet we need to submit model to pytorch hub



## Model Architecture 🧠🔧

```python
model = TransformerWrapper(
        num_tokens=64007,
        max_seq_len=8192,
        use_abs_pos_emb=False,
        tokenizer=tokenizer, # !
        embedding_provider=AndromedaEmbedding(),
        attn_layers = Decoder(
            dim=128, # 2048
            depth=8, # 16
            dim_head=128,
            heads=8,
            alibi_pos_bias=True,
            alibi_num_heads=4,
            rotary_xpos=True,
            attn_flash = True,
            deepnorm=True,
            shift_tokens=1,
            attn_one_kv_head = True,
            qk_norm=True,
            attn_qk_norm=True,
            attn_qk_norm_dim_scale=True # set this to True, in addition to `attn_qk_norm = True`
        )
    )
```

## Roadmap 🗺️📍

1. **Training phase**: Train Phi on a large-scale dataset to achieve SOTA performance in various natural language processing tasks.

2. **World-class inference infrastructure**: Establish a robust and efficient infrastructure that leverages techniques such as:

   - Model quantization: Reduce memory and computational requirements without significant loss in performance.
   - Distillation: Train smaller, faster models that retain the knowledge of the larger model.
   - Optimized serving frameworks: Deploy Phi using efficient serving frameworks, such as NVIDIA Triton or TensorFlow Serving, for rapid inference.

3. **Continuous improvement**: Continuously fine-tune Phi on diverse data sources and adapt it to new tasks and domains.

4. **Community-driven development**: Encourage open-source contributions, including pre-processing improvements, advanced training techniques, and novel use cases.

## Why Phi? 🌠💡

Phi can potentially be finetuned with 100k+ token sequence length.
Phi is a state-of-the-art language model that leverages advanced techniques to optimize its performance and efficiency. Some of these techniques include alibi positional bias, rotary position encodings (xpos), flash attention, and deep normalization (deepnorm). Let's explore the benefits of these techniques and provide some usage examples.

### Alibi Positional Bias

Alibi positional bias allows the model to learn relative positions between tokens, enabling it to better capture the relationships and dependencies between tokens in a sequence.

Usage example:

```python
attn_layers = Decoder(
    ...
    alibi_pos_bias=True,
    alibi_num_heads=4,
    ...
)
```

### Rotary Position Encodings (xpos)

Rotary position encodings introduce a more efficient way to encode positions in the input sequence. They avoid the need for absolute positional embeddings, reducing the model's memory footprint and improving training speed.

Usage example:

```python
attn_layers = Decoder(
    ...
    rotary_xpos=True,
    ...
)
```

### Flash Attention

Flash attention speeds up the self-attention mechanism by reducing the number of attention computations. It accelerates training and inference while maintaining a high level of performance.

Usage example:

```python
attn_layers = Decoder(
    ...
    attn_flash=True,
    ...
)
```

Usage example:

```python
attn_layers = Decoder(
    ...
    deepnorm=True,
    ...
)
```

### Deep Normalization (deepnorm)

Deep normalization is a technique that normalizes the activations within a layer, helping with training stability and convergence. It allows the model to better learn complex patterns and generalize to unseen data.

# Phi Principles
- **Efficiency**: Phi incorporates cutting-edge optimization techniques, such as attention flashing, rotary position encodings, and deep normalization, resulting in efficient training and inference.

- **Flexibility**: The modular design of Phi allows for easy adaptation to various tasks and domains, making it a versatile choice for a wide range of applications.

- **Scalability**: Phi's architecture is designed to scale with the ever-growing computational resources and data sizes, ensuring its continuous relevance in the NLP landscape.

- **Community-driven**: As an open-source project, Phi thrives on contributions from the community, fostering an environment of collaboration, innovation, and continuous improvement.

Join us on this exciting journey to create a powerful, efficient, and intelligent language model that will revolutionize the NLP landscape! 🚀🌟

## Todo:

* [Pretrain on Falcon](https://huggingface.co/datasets/tiiuae/falcon-refinedweb)

* [Finetune on this](https://huggingface.co/datasets/Open-Orca/OpenOrca)

* [Create synthetic datasets with the Distiller](https://github.com/Agora-X/The-Distiller)

# Implementing the Phi-1 Model

This guide is meant to assist you in implementing our Phi-1 model based on the decoder-only transformer model [VSP+ 17] using the FlashAttention implementation of multihead attention (MHA) [DFE+ 22].

## 1. Architecture

1. **Phi-1 model**: Implement an architecture with the following specifications:
   - 24 layers
   - Hidden dimension of 2048
   - MLP-inner dimension of 8192
   - 32 attention heads of dimension 64 each
2. **Phi1-small model**: Implement an architecture with the following specifications:
   - 20 layers
   - Hidden dimension of 1024
   - MLP-inner dimension of 4096
   - 16 attention heads of dimension 64 each
3. For both architectures, include rotary position embedding [SLP+ 21] with a rotary dimension of 32.
4. Tokenize your data using the same tokenizer as codegen-350M-mono [NPH+ 22].

## 2. Pretraining

1. Concatenate your dataset into a single dimensional array, using the "⟨∣endoftext∣⟩" token for separating files.
2. Train your model on a sequence length of 2048 sliced from your dataset array with next-token prediction loss.
3. Utilize the AdamW optimizer and a linear-warmup-linear-decay learning rate schedule.
4. Use attention and residual dropout of 0.1.
5. Execute your training on 8 Nvidia-A100 GPUs using deepspeed.
6. Use the following specifications for training:
   - Effective batch size: 1024
   - Maximum learning rate: 1e-3
   - Warmup over 750 steps
   - Weight decay: 0.1
7. Run your training for a total of 36,000 steps, using the checkpoint at 24,000 steps as your Phi-1-base.

## 3. Finetuning

1. Finetune your Phi-1-base model on your respective finetuning dataset.
2. Follow the same setup as pretraining, but with different hyperparameters:
   - Effective batch size: 256
   - Maximum learning rate: 1e-4
   - Warmup over 50 steps
   - Weight decay: 0.01
3. Run your training for a total of 6,000 steps and pick the best checkpoint (saved every 1000 steps).
