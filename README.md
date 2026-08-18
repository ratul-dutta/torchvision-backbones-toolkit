# TorchVision Backbones Toolkit

PyTorch utilities for working with pretrained **VGG19**, **ResNet18**, and **ResNet34** backbones — covering out-of-the-box ImageNet inference, custom classification heads for transfer learning, and embedding extraction for downstream tasks such as retrieval or image captioning.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Usage](#usage)
  - [1. Pretrained Inference (ImageNet-1k)](#1-pretrained-inference-imagenet-1k)
  - [2. Custom Classification Head (VGG19)](#2-custom-classification-head-vgg19)
  - [3. Custom Classification Head (ResNet18 / ResNet34)](#3-custom-classification-head-resnet18--resnet34)
  - [4. Embedding Extraction (VGG19)](#4-embedding-extraction-vgg19)
  - [5. Embedding Extraction (ResNet18 / ResNet34)](#5-embedding-extraction-resnet18--resnet34)
- [Output Shape Reference](#output-shape-reference)
- [Installation](#installation)
- [Requirements](#requirements)
- [Use Cases](#use-cases)
- [Author](#author)
- [License](#license)

---

## Overview

This repository is a small, focused toolkit rather than a single model. It wraps three commonly used `torchvision` backbones — **VGG19**, **ResNet18**, and **ResNet34** — into three reusable patterns:

1. **Inference** with the original ImageNet-1k weights and human-readable class labels.
2. **Transfer learning**, by swapping the final classification layer for a custom number of classes.
3. **Embedding extraction**, by stripping the classifier entirely and exposing the pooled feature vector, useful as input to downstream models (e.g. captioning, retrieval, clustering).

All examples run on dummy input tensors of shape `(B, 3, 224, 224)`, so you can verify shapes and wiring before plugging in a real dataset.

---

## Features

- Drop-in inference for VGG19 / ResNet18 / ResNet34 with `DEFAULT` pretrained weights
- Top-5 class prediction printout using `torchvision`'s bundled ImageNet category metadata
- Custom classification head replacement for arbitrary `num_classes`
- Dedicated embedding-extractor classes (`VGG19Embedding`, `ResNetEmbedding`) for feature extraction
- Automatic CUDA/CPU device selection
- No dataset required to try it out — everything runs on random dummy tensors

---

## Project Structure

```text
torchvision-backbones-toolkit/
│
├── README.md
├── requirements.txt
├── inference_pretrained.py       # Section 1: raw ImageNet inference + top-5 predictions
├── modify_vgg19_head.py          # Section 2: custom classifier head for VGG19
├── modify_resnet_head.py         # Section 3: custom classifier head for ResNet18/34
├── vgg19_embedding_extractor.py  # Section 4: VGG19 feature/embedding extractor
└── resnet_embedding_extractor.py # Section 5: ResNet18/34 feature/embedding extractor
```

> If you're keeping this as a single notebook instead of split scripts, name it `backbones_toolkit.ipynb` and keep the section headers as markdown cells matching the sections above.

---

## Model Details

| Model     | Pretrained Weights     | Input Size | Classifier Input Dim | Embedding Dim |
|-----------|--------------------------|------------|------------------------|----------------|
| VGG19     | `VGG19_Weights.DEFAULT`     | 224×224×3  | 4096                    | 25088 (512×7×7) |
| ResNet18  | `ResNet18_Weights.DEFAULT`  | 224×224×3  | 512                     | 512            |
| ResNet34  | `ResNet34_Weights.DEFAULT`  | 224×224×3  | 512                     | 512            |

---

## Usage

### 1. Pretrained Inference (ImageNet-1k)

Runs all three models on a dummy batch, prints logits/probability shapes, and reports top-5 predicted classes per sample using the weights' bundled category metadata.

```bash
python inference_pretrained.py
```

Core pattern:

```python
weights = VGG19_Weights.DEFAULT
model = vgg19(weights=weights).to(device).eval()

with torch.no_grad():
    logits = model(dummy_x)
    probs = F.softmax(logits, dim=1)
```

### 2. Custom Classification Head (VGG19)

Replaces VGG19's final `Linear(4096, 1000)` layer with `Linear(4096, num_classes)` for transfer learning.

```bash
python modify_vgg19_head.py
```

```python
in_features = model_vgg.classifier[6].in_features  # 4096
model_vgg.classifier[6] = nn.Linear(in_features, num_classes)
```

### 3. Custom Classification Head (ResNet18 / ResNet34)

Same idea, but ResNet exposes its final layer as `model.fc` instead of an indexed classifier block.

```bash
python modify_resnet_head.py
```

```python
in_features = model_res18.fc.in_features
model_res18.fc = nn.Linear(in_features, num_classes)
```

### 4. Embedding Extraction (VGG19)

`VGG19Embedding` keeps only the convolutional `features` and `avgpool` stages, dropping the classifier, and returns a flattened 25088-dim embedding per image.

```bash
python vgg19_embedding_extractor.py
```

```python
class VGG19Embedding(nn.Module):
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)  # (B, 25088)
```

### 5. Embedding Extraction (ResNet18 / ResNet34)

`ResNetEmbedding` rebuilds the ResNet stem and residual stages manually, keeping the global average pool but removing `fc`, to return a compact 512-dim embedding.

```bash
python resnet_embedding_extractor.py
```

```python
model = ResNetEmbedding(arch="resnet18").to(device).eval()
embedding = model(dummy_x)  # (B, 512)
```

---

## Output Shape Reference

| Script                              | Output              | Shape (batch=2 or 4) |
|--------------------------------------|----------------------|------------------------|
| `inference_pretrained.py`           | VGG19 logits         | (2, 1000)              |
| `inference_pretrained.py`           | ResNet18/34 logits   | (2, 1000)              |
| `modify_vgg19_head.py`              | Custom VGG19 logits  | (4, num_classes)       |
| `modify_resnet_head.py`             | Custom ResNet logits | (4, num_classes)       |
| `vgg19_embedding_extractor.py`      | VGG19 embedding      | (2, 25088)             |
| `resnet_embedding_extractor.py`     | ResNet18/34 embedding| (2, 512)                |

---

## Installation

```bash
git clone https://github.com/ratul-dutta/torchvision-backbones-toolkit.git
cd torchvision-backbones-toolkit
pip install -r requirements.txt
```

Or install dependencies directly:

```bash
pip install torch torchvision
```

---

## Requirements

- Python 3.8+
- PyTorch
- Torchvision
- CUDA-enabled GPU (optional, falls back to CPU automatically)

---

## Use Cases

- **Quick sanity checks** on pretrained backbone shapes before building a larger pipeline
- **Transfer learning starter code** for a new classification task with a small number of classes
- **Feature/embedding extraction** for downstream tasks such as image captioning, image retrieval, clustering, or as frozen input features to another model

---

## Author

**Ratul Dutta**
M.Sc. Mathematics & Computing, Banaras Hindu University
[GitHub](https://github.com/ratul-dutta) · [LinkedIn](https://www.linkedin.com/in/ratul-dutta-33441b317)

---

## License

This project is available under the [MIT License](LICENSE).
