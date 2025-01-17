# Transformer Model Implementation 🤖

## Overview

This repository contains an implementation of a Transformer model for sequence-to-sequence tasks. The model follows the architecture described in the "Attention Is All You Need" paper and is designed for multilingual translation using the OpusBooks dataset.

## Architecture Details

The Transformer processes data through multiple steps:

### Data Flow
1. **Input Processing**
   - Input sequence → Embeddings (with Position Encoding)
   - Embeddings → Encoder

2. **Encoding Stage**
   - Encoder stack processes input
   - Produces encoded representation

3. **Decoding Stage**
   - Target sequence preparation (start-of-sentence token added)
   - Conversion to Embeddings (with Position Encoding)
   - Decoder processing

4. **Output Generation**
   - Decoder stack processes with encoder's output
   - Generates encoded target sequence
   - Output layer → word probabilities → final sequence

5. **Training**
   - Loss calculation: output vs target sequence
   - Gradient generation
   - Back-propagation training

## Core Components

### Encoder Block
- Multi-Head Attention mechanism
- Feed-Forward network (fully connected)
- Residual connections around sub-layers
- Layer normalization
- Output dimension: d_model=512

### Decoder Block
- Extended encoder architecture with:
  - Additional multi-head attention layer for encoder output
  - Modified self-attention (masked)
  - Position-aware processing
  - Prevention of future position attention

## Dataset Information

I used the OpusBooks dataset from 🤗 Hugging Face:

```python
Features:
- id: unique identifier
- translation: paired sentences in multiple languages
```

Supported language pairs include:
- Spanish ↔️ Portuguese
- English ↔️ French
- Other language combinations

## Model Parameters

```python
MODEL_PARAMS = {
    'd_model': 512,        # Model dimension
    'n_heads': 8,          # Number of attention heads
    'n_layers': 6,         # Number of encoder/decoder layers
    'dropout': 0.1         # Dropout rate
}
```

## Requirements

```bash
torch>=1.8.0
transformers>=4.5.0
datasets>=1.6.0
```

## Usage

```python
from transformer import Transformer

# Initialize model
model = Transformer(
    src_vocab_size=32000,
    tgt_vocab_size=32000,
    d_model=512
)

# Training
model.train()

# Inference
model.translate(source_text)
```
