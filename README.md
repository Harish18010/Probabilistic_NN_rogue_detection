# LoRaGuard: Probabilistic Adversarial RF Device Authentication

Course project for EE337, IIT Guwahati. Report: "Probabilistic ResNet with Self-Attention for Adversarial LoRa Fingerprinting."

## Overview

Deep learning classifiers used for LoRa Radio Frequency Fingerprinting (RFFI) fail against zero-day spoofing attacks because a softmax output layer forcefully maps unknown signals to known device classes. This project addresses that failure mode in two phases:

1. **Baseline uncertainty estimation.** A Probabilistic Multi-Task ResNet with self-attention is trained on 16,384-point LoRa I/Q signals reshaped as 128x128 single-channel images. Monte Carlo Dropout (T = 10 stochastic passes) is used to estimate epistemic uncertainty at inference. This baseline detects only 30% of rogue signals; the remaining 70% evade detection because their latent representations overlap with known device clusters.
2. **Adversarial hardening and feature-space detection.** FGSM adversarial fine-tuning (epsilon = 0.05) is applied not to improve classification accuracy, but as a geometric regularizer that tightens intra-class latent clusters. The softmax head is then discarded entirely in favor of a 1-Nearest-Neighbor anomaly detector operating on 64-dimensional feature vectors extracted after the attention and pooling layers. This produces clean, non-overlapping separation between legitimate and rogue signals, achieving 0.82 ROC-AUC with an inference latency of 1.68 ms.

## Architecture

Input (128x128 I/Q image) -> 7x7 stem convolution (32ch, stride 2, BatchNorm, ReLU) -> three residual blocks (32, 64, 64 channels) -> channel-wise self-attention -> global average pooling (64-dim feature vector) -> MC Dropout (p = 0.3) -> two linear heads (30-way device identification, binary rogue detection).

At inference, the softmax heads are bypassed for anomaly detection; only the 64-dim feature vector is used, compared against a reference database of legitimate devices via nearest-neighbor distance.

## Repository Structure

- `config.py` — Global hyperparameters and thresholds
- `explore_data.py` — Dataset inspection utility
- `requirements.txt` — Python dependencies
- `src/`
  - `dataset.py` — LoRaDataset loader for .h5 I/Q signal files
  - `models/`
    - `resnet_base.py` — Residual block definitions
    - `attention.py` — Channel-wise self-attention module
    - `probabilistic.py` — ProbabilisticMultiTaskResNet (full model)
  - `train.py` — Baseline training (device ID + rogue detection heads)
  - `train_adv.py` — FGSM adversarial fine-tuning
  - `test_rogue.py` — MC Dropout uncertainty evaluation on rogue signals
  - `test_anomaly.py` — 1-NN feature-space anomaly detection
  - `evaluate.py` — Evaluation utilities
  - `visualize.py` — Distance histogram and t-SNE plot generation

## Method Summary

**Self-attention.** A channel-wise self-attention module is blended into the residual feature maps via a learned scalar gate (initialized to zero), allowing the network to gradually incorporate global context during training.

**MC Dropout uncertainty.** With dropout active at inference, T = 10 stochastic forward passes yield a predictive mean and variance per signal. Signals with variance above a fixed threshold are flagged as anomalous under the baseline approach.

**FGSM adversarial fine-tuning.** Adversarial examples are crafted via the fast gradient sign method and used to fine-tune the pretrained model for 5 epochs, restructuring the latent space so that unknown signals are pushed geometrically outside all known device regions.

**1-NN feature-space detection.** After adversarial hardening, raw 64-dim feature vectors are extracted and compared against a database of 100 legitimate reference vectors using Euclidean distance. A fixed threshold (tau = 0.80) separates legitimate from rogue signals with no overlap between the two populations in evaluation.

## Setup

```bash
pip install -r requirements.txt
```

Dependencies: PyTorch, torchvision, NumPy, h5py, Matplotlib, scikit-learn, tqdm.

## Usage

Train the baseline multi-task model:
```bash
python src/train.py
```

Run FGSM adversarial fine-tuning on a pretrained checkpoint:
```bash
python src/train_adv.py
```

Evaluate MC Dropout uncertainty on rogue signals:
```bash
python src/test_rogue.py
```

Run 1-NN feature-space anomaly detection:
```bash
python src/test_anomaly.py
```

Generate the distance histogram and t-SNE visualizations:
```bash
python src/visualize.py
```

## Results

- Baseline MC Dropout rogue detection rate: 30% (70% evasion due to overlapping latent clusters)
- Post-hardening 1-NN anomaly detection: 0.82 ROC-AUC, 1.68 ms inference latency
- Legitimate and rogue signal distributions separate cleanly at threshold tau = 0.80

## Key Insight

Classification accuracy is not a reliable objective for open-set security. Geometric control over the latent feature space, enforced via adversarial fine-tuning and evaluated with a simple nearest-neighbor rule, provides a more robust basis for zero-day device detection than softmax-based uncertainty alone.

## Future Work

- Evaluate Mahalanobis distance in place of Euclidean 1-NN distance
- Test robustness under real-world multipath channel conditions


## References
[1] G. Shen, J. Zhang, A. Marshall, and J. R. Cavallaro, "Towards scalable and channel-robust radio frequency fingerprint identification for LoRa," IEEE Trans. Inf. Forensics Security, vol. 17, pp. 774-787, 2022.

[2] Y. E. Sagduyu and T. Erpek, "Adversarial attack and defense for LoRa device identification and authentication via deep learning," 2024.

This project is done for educational purposes.


