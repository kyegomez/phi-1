# phi-1
Plug in and play implementation of " Textbooks Are All You Need", ready for training, inference, and dataset generation

# Model Card

## Model Description

This model is a decoder-only transformer based on the FlashAttention implementation of multihead attention (MHA) [DFE+22]. It follows a parallel configuration, similar to recent models such as CodeGen [NPH+22], PaLM [CND+22], and GPT-NeoX [BBH+22], using MHA and MLP layers. The architecture consists of 24 layers for the 1.3 billion parameter phi-1 model and 20 layers for the 350 million parameter phi1-small model. The hidden dimension is 2048 for phi-1 and 1024 for phi1-small, with an MLP-inner dimension of 8192 and 4096, respectively. Both models have attention heads of dimension 64. The rotary position embedding with a rotary dimension of 32, introduced in [SLP+21], is utilized. The architectural choices are adopted from [NPH+22]. The tokenizer used is the same as codegen-350M-mono [NPH+22]. Notable techniques such as Fill-In-the-Middle (FIM) [BJT+22] or Multi-Query-Attention (MQA) [RSR+20] are not incorporated, which could further enhance performance and efficiency [LAZ+23].


## Model Architecture Components

### Multihead Attention (MHA)
The model utilizes the FlashAttention implementation of multihead attention (MHA) [DFE+22]. MHA allows the model to attend to different parts of the input sequence simultaneously, capturing both local and global dependencies. It splits the input into multiple heads, performs separate attention calculations for each head, and then combines the results. Each head has its own learned weights, enabling the model to focus on different aspects of the input.

### MLP Layers
The model incorporates MLP (Multi-Layer Perceptron) layers in parallel configuration alongside the MHA layers. MLP layers introduce non-linearities and enable the model to capture complex relationships within the input data. By combining MHA and MLP layers, the model can effectively learn both local and global dependencies and capture intricate patterns in the data.

### Rotary Position Embedding
To incorporate positional information, the model utilizes rotary position embedding [SLP+21]. This technique introduces an additional dimension called the rotary dimension. It allows the model to learn positional encodings in a continuous manner, improving its ability to generalize across varying sequence lengths. The rotary position embedding enables the model to understand the relative positions of tokens within the input sequence.

### Decoder-Only Transformer
The model follows a decoder-only transformer architecture, which means it lacks an encoder component. The decoder is responsible for generating output sequences based on the encoded input and learned attention patterns. This architecture is commonly used in tasks where auto-regressive generation is required, such as language modeling and text generation.

### Hidden Dimensions and Attention Heads
The phi-1 model has a hidden dimension of 2048, while the phi1-small model has a hidden dimension of 1024. The hidden dimension determines the capacity of the model to capture complex representations of the input data.

Both models employ attention heads with a dimension of 64. Attention heads allow the model to focus on different aspects of the input sequence simultaneously. By using multiple attention heads, the model can learn diverse and complementary representations, enhancing its ability to capture intricate patterns and dependencies.

### Number of Layers
The phi-1 model consists of 24 layers, while the smaller phi1-small model has 20 layers. The number of layers influences the depth and expressive power of the model. Deeper models can capture more intricate dependencies but may require more computational resources.

### MLP-Inner Dimension
The MLP-inner dimension is 8192 for phi-1 and 4096 for phi1-small. The MLP-inner dimension defines the dimensionality of the hidden layers within the MLP component. It determines the model's capacity to model complex relationships and nonlinearities within the data.

By combining these various components, the model can effectively capture and encode the input sequence's information, allowing for high-quality generation and understanding of the task at hand.



## Model Training

- **Pretraining Dataset**: The model is pretrained on the CodeTextbook dataset, which includes a filtered code-language corpus and synthetic textbooks.
- **Finetuning Dataset**: The model is further finetuned on the CodeExercises dataset.

### Pretraining Details

- **Training Method**: The respective datasets are concatenated into a single dimensional array using the "⟨∣endoftext∣⟩" token as a separator.
- **Sequence Length**: The models are trained on a sequence length of 2048, sliced from the dataset array, with a next-token prediction loss.
- **Training Hardware**: The model is trained on 8 Nvidia-A100 GPUs using deepspeed.
- **Optimization**: The training utilizes fp16 training with the AdamW optimizer and a linear-warmup-linear-decay learning rate schedule.
- **Dropout**: Attention and residual dropout of 0.1 is applied.
- **Training Time**: The pretrained base model, phi-1-base, is obtained in under 4 days of training.

### Finetuning Details

- **Finetuning Method**: Finetuning is performed on phi-1-base using the CodeExercises dataset.
- **Hyperparameters**: The same setup as pretraining is used, but with different hyperparameters. The effective batch size is 256, maximum learning rate is 1e-4 with 50 steps of warmup, and weight decay is 0.01.
- **Training Time**: Finetuning to obtain phi-1 requires an additional 7 hours on the same hardware.

## Performance

- **Pretraining Accuracy**: phi-1-base achieves a 29% accuracy on the HumanEval dataset.
- **Evaluation Metrics**: Further evaluation metrics are not provided in the given model description.


## References

- [VSP+17] Reference to the decoder-only

 transformer paper used as a model.
- [DFE+22] Reference to the FlashAttention implementation of multihead attention.
- [NPH+22] Reference to the CodeGen paper.
- [CND+22] Reference to the PaLM paper.
- [BBH+22] Reference to the GPT-NeoX paper.
- [SLP+21] Reference to the rotary position embedding paper.
- [BJT+22] Reference to the Fill-In-the-Middle (FIM) technique.
- [RSR+20] Reference to the Multi-Query-Attention (MQA) technique.
- [LAZ+23] Reference to the paper mentioning techniques that could enhance performance and efficiency.

(Note: The references provided above are placeholders and need to be replaced with the actual references from the model's source documentation.)