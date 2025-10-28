import torch
import time
from flash import TritonAttention

def benchmark(seq_len=1024, head_dim=64, num_heads=8, causal=True):
    torch.manual_seed(0)
    Q = torch.randn(1, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)
    scale = 1.0 / (head_dim ** 0.5)

    print(f"\nBenchmarking FlashAttention vs PyTorch SDPA (seq_len={seq_len})")

    # PyTorch native attention
    torch.cuda.synchronize()
    start = time.time()
    out_ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=causal)
    torch.cuda.synchronize()
    print(f"PyTorch SDPA time: {time.time() - start:.4f}s")

    # Triton FlashAttention
    torch.cuda.synchronize()
    start = time.time()
    out_flash = TritonAttention.apply(Q, K, V, causal, scale)
    torch.cuda.synchronize()
    print(f"Triton FlashAttention time: {time.time() - start:.4f}s")

    diff = (out_ref - out_flash).abs().max().item()
    print(f"Max difference: {diff:.6f}")

if __name__ == "__main__":
    benchmark(512)
    benchmark(1024)
    benchmark(2048)
