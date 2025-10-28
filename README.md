# FlashAttention-Learn 

A minimal and educational implementation of **FlashAttention v2** in Triton,  
based on the original work by **Tri Dao (Stanford / Together.ai)**.

---

## Overview
This repo provides a standalone `flash.py` kernel implementing forward and backward
FlashAttention using pure **Triton** (no C++ or CUDA bindings).

## Features
- Full forward + backward pass
- Causal and non-causal attention
- Benchmark comparison with PyTorch SDPA
- Minimal dependencies (Torch + Triton)

---

## Folder Structure

```bash
flash-attn-learn/
├── flash.py 
├── benchmark.py 
├── README.md
├── requirements.txt
└── LICENSE


```
## Usage
```bash
pip install -r requirements.txt
python benchmark.py