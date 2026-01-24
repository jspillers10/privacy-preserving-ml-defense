# HTB Machine Learning Security: Privacy Defense Challenge

## Challenge Overview

**Objective**: Train a privacy-preserving Fashion-MNIST classifier that resists membership inference attacks (MIA) while maintaining utility.

**Requirements**:
- Test accuracy ≥ 70%
- MIA vulnerability reduction ≥ 40%
- Use exact FashionMNISTCNN architecture

## Solution Approach

### Defense Strategy: Differential Privacy with DP-SGD

Implemented **Differentially Private Stochastic Gradient Descent** using the Opacus library to provide formal privacy guarantees.

**Key Mechanisms**:
1. **Per-sample gradient clipping** - Bounds individual sample influence
2. **Gaussian noise injection** - Masks individual contributions
3. **Privacy accounting** - Tracks cumulative privacy budget

### Configuration

```python
EPSILON = 3.0              # Privacy budget
DELTA = 1e-5               # Privacy failure probability
MAX_GRAD_NORM = 1.0        # Gradient clipping threshold
NOISE_MULTIPLIER = 1.1     # Gaussian noise scale
EPOCHS = 20
BATCH_SIZE = 256           # Larger batch = better privacy
LEARNING_RATE = 0.001
```

## Results

| Metric | Requirement | Achieved | Status |
|--------|-------------|----------|--------|
| Test Accuracy | ≥70% | **81.2%** | +11.2% |
| MIA Improvement | ≥40% | **93.02%** | +132% |
| MIA Advantage | Reduce | **0.0062** | (from 0.0895) |
| Overall Score | - | **87.11/100** | DONE |

**Flag**: `HTB{pr*****_d*******_m*****_c*******}`

### Privacy Analysis

- **Baseline MIA Accuracy**: 54.5% (attacker has 9% advantage)
- **Defended MIA Accuracy**: 50.62% (attacker has 0.62% advantage)
- **Interpretation**: Attack success reduced to near-random guessing

The model achieves **(ε=3.0, δ=1e-5)-differential privacy**, providing formal mathematical guarantees that individual training samples cannot be reliably identified.

## Technical Implementation

### Architecture (FashionMNISTCNN)

```
Input (28×28×1)
    ↓
Conv2d(1→32) + ReLU + MaxPool
    ↓
Conv2d(32→64) + ReLU + MaxPool
    ↓
Conv2d(64→64) + ReLU + MaxPool
    ↓
Flatten → FC(576→128) + ReLU
    ↓
FC(128→10) → Output
```

**Parameters**: 122,858 trainable parameters

### Data Preprocessing

Critical: Match server's normalization exactly
```python
transforms.Normalize((0.2860,), (0.3530,))
```

## Quick Start

### Installation

```bash
# CPU-only version (recommended for space constraints)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu --break-system-packages
pip3 install opacus safetensors --break-system-packages
```

### Training

```bash
python3 train.py
```

Training time: ~15-20 minutes on CPU, ~5-10 minutes on GPU

### Submission

```bash
export BASE_URL="http://target_ip:port"
curl -X POST "$BASE_URL/submit" -F "defended_model=@defended_model.safetensors"
```

## Key Learnings

### Why DP-SGD Works

1. **Gradient Clipping**: Prevents any single sample from having outsized influence on model updates
2. **Noise Addition**: Provides plausible deniability - can't determine if specific sample was in training set
3. **Privacy Accounting**: Mathematically proven bounds on information leakage

### Privacy-Utility Tradeoff

- **Accuracy cost**: Minimal (81.2% vs ~82% baseline)
- **Privacy gain**: Massive (93% vulnerability reduction)
- **Ratio**: ~6200:1 privacy gained per utility lost

### Membership Inference Attacks

**How MIA Works**:
1. Attacker observes prediction confidence
2. Training samples often have higher confidence
3. Attack model learns to distinguish these patterns

**Defense Effectiveness**:
- Confidence distributions now overlap
- Attack accuracy drops to ~50% (random guessing)
- Formal privacy guarantee: (ε=3.0, δ=1e-5)

## Real-World Applications

This privacy-preserving approach is critical for:

- **Healthcare**: Medical imaging without patient data exposure
- **Finance**: Fraud detection preserving transaction privacy
- **Enterprise**: User analytics with GDPR/CCPA compliance
- **Government**: Sensitive data analysis with privacy guarantees

## Files Included

- `model.py` - CNN architecture definition
- `train.py` - DP-SGD training script
- `README.md` - This writeup

## References

- [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133) - Abadi et al., 2016
- [Membership Inference Attacks](https://arxiv.org/abs/1610.05820) - Shokri et al., 2017
- [Opacus Library](https://opacus.ai/) - PyTorch Differential Privacy

## Author

**Jake Spillers**  
Cyber Security Specialist @ Live Nation Entertainment  
Focus: AI/ML Security, Application Security, Third-Party Risk

---

**Challenge**: HackTheBox Machine Learning Security Module  
**Difficulty**: Medium  
**Category**: Privacy-Preserving Machine Learning  
**Completion Date**: January 2026
