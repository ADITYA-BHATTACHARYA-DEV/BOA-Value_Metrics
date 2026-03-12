# HybridPillar++ Compression Architecture: For HEP Compression- A Proposed System Procedure
<img width="1350" height="810" alt="image" src="https://github.com/user-attachments/assets/bced7842-2ec3-4d49-a744-5336f1e28398" />

---
---
This document explains the **core techniques and algorithms** used in the HybridPillar++ neural compression system designed for **High-Energy Physics (HEP) byte-stream data**. The architecture combines neural modeling with classical compression theory to achieve **high compression ratios, high throughput, and bit-exact reconstruction**.

---


# Installation Guide

This section explains how to **set up the environment and install the dependencies** required to run the **HybridPillar++ / BOA Constrictor compression system**.

The following steps cover:

- cloning the repository
- configuring the Python environment
- installing deep learning frameworks
- installing compression libraries
- enabling GPU acceleration
- installing profiling tools

---

## 1. Clone the Repository

First, download the project repository from GitHub.

```bash
git clone https://github.com/ADITYA-BHATTACHARYA-DEV/boa-constrictor.git
cd boa-constrictor
```
- This repository contains:
- neural compression models
- diagnostic tools
- training scripts
- range coder backend
- benchmarking utilities


## 2. System & Hardware Prerequisites

The HybridPillar++ architecture relies on GPU acceleration for high-speed inference.

Required Hardware

NVIDIA GPU (recommended)

At least 8 GB GPU memory for efficient training
| Requirement    | Description                               |
| -------------- | ----------------------------------------- |
| CUDA Toolkit   | Version **11.8** or **12.1+** recommended |
| cuDNN          | Must match the installed CUDA version     |
| NVIDIA Drivers | Updated drivers supporting CUDA           |

- These components enable CUDA kernel execution for Mamba and range coding.
---

## 3. Core Python Environment

It is recommended to use Python 3.10+ inside a dedicated Conda environment to avoid dependency conflicts.

- Create Environment
```bash
conda create -n hybrid_boa python=3.10
```

- Activate Environment
```bash
conda activate hybrid_boa
```
This ensures all required libraries are isolated within the project environment.

## 4. Deep Learning Frameworks

 The compression model is built using PyTorch with CUDA support.

- Install PyTorch using the official CUDA-enabled wheel.
```bash 
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
### This installation provides:

- GPU acceleration

- tensor operations

- neural network layers

- CUDA runtime support

## 5. Mamba & Selective State Space Requirements

The Selective State Space Model (SSM) used in HybridPillar++ requires specialized CUDA kernels.

- Install Kernel Compilation Tools
```bash 
pip install ninja packaging
```

- These packages allow just-in-time compilation of CUDA kernels.

- Install Causal Convolution Layer

- Causal convolution is required for streaming sequence modeling.
```bash
pip install causal-conv1d>=1.2.0
```
- Install Mamba State Space Model
```bash
pip install mamba-ssm
```
### This library provides:

- selective state space layers

- efficient long-sequence processing

- linear-time scaling for sequence modeling

## 6. Compression & Scientific Libraries

- Standard scientific libraries are required for:

- numerical computations

- signal processing

- diagnostics

- configuration parsing

### Install the following packages:
```bash
pip install numpy scipy matplotlib pyyaml
```

| Package    | Purpose                            |
| ---------- | ---------------------------------- |
| numpy      | Numerical computations             |
| scipy      | FFT analysis and signal processing |
| matplotlib | Visualization of diagnostics       |
| pyyaml     | Configuration file parsing         |


---
## 7. Sustainability & Profiling Tools

HybridPillar++ measures energy efficiency and environmental impact.

### Install Carbon Footprint Monitoring
```bash
pip install codecarbon
```
### This tool tracks:

- GPU power consumption

- carbon emissions

- energy efficiency metrics

### Install Progress Monitoring
```bash
pip install tqdm
```
### This provides real-time progress bars during:

- training

- evaluation

- compression benchmarking

## 8.Range Coder Backend

If using a Python-based Range Coder implementation, additional libraries may be required.

### Install Bitstream Library
```bash
pip install bitstream
```
### This package provides:

- bit-level operations

- entropy coding utilities

- binary stream manipulation

### Range coding is responsible for converting model probability predictions into compressed bitstreams.
---

# Environment Setup Summary

After installation, your environment should contain the following core components:


| Category              | Tools        |
| --------------------- | ------------ |
| Deep Learning         | PyTorch      |
| Sequence Modeling     | Mamba SSM    |
| GPU Compilation       | Ninja        |
| Numerical Computing   | NumPy, SciPy |
| Visualization         | Matplotlib   |
| Configuration         | PyYAML       |
| Compression Utilities | Bitstream    |

---
## The Workflow
<img width="4881" height="7729" alt="Data Ingestion to-2026-03-12-101617" src="https://github.com/user-attachments/assets/cbab61d6-c564-452d-a9b6-9c29ef09b301" />

# Key Techniques

## 1. RMSNorm (Root Mean Square Normalization)

**What:**  
A lightweight stabilization layer that ensures neural signals remain within a manageable range.

**Why:**  
High-Energy Physics detector data often contains **extreme signal spikes from particle collisions**.  
Unlike standard normalization, which centers the data, **RMSNorm preserves the relative strength of physics hits** while preventing numerical instability.

**How:**  
It scales signals using their **root mean square energy** instead of their mean value.

**Impact:**
- Prevents exploding activations
- Preserves meaningful detector spikes
- Improves computational efficiency
- Contributes to **~960 MB/s throughput**

---

## 2. Selective State Space Pillar (Mamba-lite SSM)
<img width="527" height="647" alt="image" src="https://github.com/user-attachments/assets/3b2e3a69-55fc-4aa5-a90e-daf983c83962" />

**What:**  
A recurrent memory mechanism that maintains a **running summary of the entire data stream**.

**Why:**  
Detector streams contain persistent **background noise (pedestal signals)**.  
The SSM learns to identify and ignore this background so the model can focus on **true particle events**.

**How:**  
- Maintains a **fixed-size hidden state**
- Uses **selective gating** to decide what information to retain or forget

**Impact:**
- Provides **global context**
- Improves prediction accuracy
- Reduces wasted compression bits

---

## 3. Sliding Window Attention Pillar

**What:**  
A localized attention mechanism that focuses only on the **most recent 128 bytes**.

**Why:**  
Particle jets occur in **short bursts of energy**.  
Long-distance context rarely helps predict immediate detector signals.

**How:**  
- Computes attention only within a **fixed 128-byte window**
- Ignores distant history to avoid expensive memory costs

**Impact:**
- Captures **local particle interactions**
- Maintains high prediction precision
- Avoids quadratic attention scaling

---
<img width="367" height="262" alt="image" src="https://github.com/user-attachments/assets/a3e2d2ee-e318-4c84-b110-09447e871147" />

## 4. KV (Key-Value) Cache Layer

**What:**  
A temporary cache storing attention information from previous tokens.

**Why:**  
In **byte-by-byte autoregressive compression**, each prediction would otherwise recompute the previous 128 bytes.

**How:**  
- Stores computed keys and values
- Adds new tokens while removing the oldest

**Impact:**
- Major driver of **inference speed**
- Enables **near 1 GB/s processing rate**

---

## 5. Grouped 1D-Convolution Pillar

**What:**  
A convolutional layer that detects **repeating structural patterns**.

**Why:**  
Physics data often includes:
- Hardware markers
- Detector synchronization signals
- Configuration headers

These simple structures should not consume expensive attention resources.

**How:**  
- Uses small filters sliding across the byte stream
- Uses **grouped convolution** to reduce computation

**Impact:**
- Efficient pattern recognition
- Reduces neural compute load
- Supports **sustainable GPU usage**

---

## 6. SwiGLU Gating (MLP Pillar)

**What:**  
A gated feedforward layer controlling information flow.

**Why:**  
Most detector data represents **background noise**.  
SwiGLU suppresses irrelevant signals while amplifying important physics patterns.

**How:**  
Uses two parallel paths:

1. Feature computation
2. Gating path acting as a **volume control**
<img width="546" height="172" alt="image" src="https://github.com/user-attachments/assets/60c82416-d82c-4c73-aad3-b324f2e62d31" />

The gate multiplies features to either **boost or suppress signals**.

**Impact:**
- Filters entropy noise
- Helps reach **0.92 BPB compression floor**

---

## 7. QuantLinear Layer (Weight Quantization)
<img width="476" height="192" alt="image" src="https://github.com/user-attachments/assets/3cc0d26b-2262-4658-b500-a7231e4154cf" />

**What:**  
A compressed neural layer using **8-bit integer weights**.

**Why:**  
Floating-point models consume large memory and power.  
Quantization allows the model to fit entirely inside **GPU cache memory (~12.2 MB)**.

**How:**  
- Uses **Quantization-Aware Training (QAT)**
- Converts weights to **INT8 integers**

**Impact:**
- Faster inference
- Lower power consumption
- Enables **hardware acceleration via Tensor Cores**

---

## 8. Asynchronous Range Coder Backend

**What:**  
A mathematical entropy encoder converting model predictions into the final compressed bitstream.

**Why:**  
Neural networks only provide **probability predictions**.  
The Range Coder converts these probabilities into **optimal bit representations**.

**How:**  
- Runs in a separate **CUDA stream**
- Neural network predicts next data block
- Range coder simultaneously compresses previous block

**Impact:**
- Ensures **bit-exact compression**
- Enables **960 MB/s throughput**

---

# Layer Importance Summary

| Pillar / Layer | Core Task | Impact on Research Goals |
|---|---|---|
| RMSNorm | Stability | Prevents crashes from large physics spikes |
| SSM (Mamba) | Global Memory | Learns background detector noise |
| Sliding Attention | Local Precision | Detects particle hit patterns |
| KV Cache | Speed | Enables ~1 GB/s throughput |
| SwiGLU | Filtering | Achieves 0.92 BPB compression |
| Quantization | Sustainability | Reduces energy and memory use |

---
<img width="2990" height="3490" alt="test_result_3" src="https://github.com/user-attachments/assets/86a67987-a4cc-4ef2-9852-9f87dfa2b3c1" />


# Base Compression Algorithms

HybridPillar++ integrates ideas from classical compression algorithms such as **ZLIB (DEFLATE)** and **LZMA**.

---

## ZLIB (DEFLATE Algorithm)

ZLIB implements the **DEFLATE compression algorithm**, widely used in:

- Gzip files
- PNG images
- ZIP archives

### Stage 1: LZ77 Dictionary Matching

The algorithm scans data using a **sliding window (~32 KB)** to detect repeated sequences.

Repeated sequences are replaced with **length-distance pointers**.

Example representation: (distance, length)

This indicates:
- how far back to look
- how many bytes to copy

---

### Stage 2: Huffman Coding

The resulting symbols are compressed using **Huffman coding**, which assigns:

- Short bit sequences → frequent symbols
- Long bit sequences → rare symbols

This minimizes the total number of bits required.

---

## LZMA (Lempel-Ziv-Markov Chain Algorithm)

LZMA is used in **7-Zip (.7z)** and achieves **higher compression ratios** than DEFLATE.

However, it requires **more RAM and computation**.

---

## Large Dictionary

LZMA supports **much larger dictionaries**:1 MB → several GB

This allows the compressor to detect patterns separated by **very large distances**.

---

## Range Coding

Instead of Huffman coding, LZMA uses a **Range Coder**, a form of **arithmetic coding**.

Key advantages:

- Can assign **fractional bits**
- Achieves compression closer to **Shannon entropy limits**

---

## Markov Chain Probability Modeling

LZMA predicts future bits using **context-dependent Markov models**.

The probability of the next symbol depends on:

- previous bits
- previously observed patterns

This adaptive modeling significantly improves compression for **structured data** such as:

- source code
- configuration files
- database dumps

---

<img 
  width="561" 
  height="1026" 
  style="display: block; margin: 0 auto;" 
  alt="Physics-Informed Byte" 
  src="https://github.com/user-attachments/assets/484b4c1d-1995-4d52-8b45-db696e755119" 
/>

---

# Compression vs Throughput Trade-Off

A key research goal is identifying the balance between:

- **Compression density (BPB)**
- **Processing throughput (MB/s)**

Even small improvements in compression can affect processing speed.

Example trade-off:

| Metric | Value |
|------|------|
| Raw Data | 8.00 BPB |
| Mamba Compression | 1.25 BPB |
| HybridPillar++ | 0.92 BPB |
| Throughput | ~960 MB/s |

---

# Efficiency Optimization Techniques

Goal:

Identify the **optimal parameter size** that maximizes compression while maintaining throughput.
<img width="1800" height="1000" alt="semi_tst_6" src="https://github.com/user-attachments/assets/80734893-ee18-40b2-9f9f-37aeeef25fd1" />

---

## Model Distillation

A larger **teacher model** trains a smaller **student model**.

Benefits:

- Transfer learned physics priors
- Reduce model size
- Maintain compression accuracy

---

## Structural Pruning

Unnecessary neural connections are removed after training.

Advantages:

- Faster inference
- Lower memory usage
- Reduced energy consumption
<img width="1655" height="616" alt="Screenshot 2026-03-12 142821" src="https://github.com/user-attachments/assets/dcdedeff-bd2a-4ab3-a292-8ec7fe6762c8" />


---

## Quantization-Aware Training

During training, the model learns to operate using **8-bit integer precision**.

Benefits:

- Smaller model footprint
- Faster GPU execution
- Lower power consumption

---
<img width="2590" height="2190" alt="test_result_8" src="https://github.com/user-attachments/assets/d4802b10-7f22-4f75-9ec0-851a234732af" />

# Sustainability Considerations

Large-scale physics experiments consume substantial computing resources.

HybridPillar++ improves sustainability through:

- reduced storage requirements
- reduced data transmission
- energy-efficient neural inference

<img width="1077" height="440" alt="Screenshot 2026-03-12 161354" src="https://github.com/user-attachments/assets/3d198d10-ed54-4c2c-9460-5ede2c7b86e9" />

---

# Final Performance Goals

| Metric | Target |
|------|------|
| Compression Entropy | 0.92 BPB |
| Storage Reduction | ~88.5% |
| Throughput | ~960 MB/s |
| Fidelity | Bit-Exact |
| Model Size | ~12.2 MB |

---
<img width="1397" height="292" alt="Screenshot 2026-03-12 133216" src="https://github.com/user-attachments/assets/c93748e4-39e8-4e9a-81f6-4575e93a10fa" />
<img width="743" height="346" alt="Screenshot 2026-03-12 111745" src="https://github.com/user-attachments/assets/5660a9c6-e357-46e7-9959-29ce0de6d6a1" />

# Conclusion

HybridPillar++ represents a **hybrid compression architecture** that combines:

- neural sequence modeling
- classical entropy coding
- physics-informed constraints
- hardware-aware optimization

This approach enables **extremely efficient compression of large scientific datasets**, making it suitable for modern **High-Energy Physics experiments and large-scale detector systems**.

---
##  Detailed WorkBook
https://docs.google.com/document/d/19zV-tJhv10rQzWw1gmcw8a8bsu0kKsi1zJucPckIJ7w/edit?usp=sharing

## 📧 Contact
For any queries or suggestions, contact:
📩 Email: adityabhattacharya3002@gmail.com  
📌 GitHub: [ADITYA BHATTACHARYA:- ]([https://github.com/ADITYA-BHATTACHARYA-DEV])(https://github.com/ADITYA-BHATTACHARYA-DEV)
