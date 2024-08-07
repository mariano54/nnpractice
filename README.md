## NNpractice

GPT2/3 training and inference code, written using pytorch,
based on Andrej Karpathy's nano-GPT videos.
Only for educational purposes.

Forward and backward passes implemented for the following:
- Linear 
- Batch Norm
- Layer Norm
- Softmax
- Multi-layer perceptron
- Relu
- Gelu
- Dropout
- Attention
- Transformer

Generation/training includes implementation of:
- Weight decay
- Gradient clipping
- AdamW
- Gradient accumulation
- Multi-GPU parallelism
- LLM generation
- Data loading


Huggingface transformers is used to download the GPT2 weights, and for the fineweb dataset.
The actual training code only relies on pytorch tensor code.

To install, run:

```bash
python -m venv ./.venv
source ./.venv/bin/activate
pip install -r requirements.txt  # or requirements-rocm.txt
```

GPT-2 weights can be loaded and used, or the model can be trained from scrath.
To load GPT2 weights, run:

```bash
python -m src.load_gpt2_weights
```

To download 10B of fineweb tokes:

```bash
python -m src.download_fineweb
```

To train the model:

```bash
python -m src.gpt False
```



