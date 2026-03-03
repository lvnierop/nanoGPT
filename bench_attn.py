#!/usr/bin/env python3
"""
Benchmark nanoGPT CausalSelfAttention speed with and without use_identity_block.

Usage:
  python bench_attn.py
  python bench_attn.py --device cuda --B 8 --T 1024 --C 768 --n_head 12 --iters 200 --warmup 50
"""

import argparse
import time
from types import SimpleNamespace

import torch

# Adjust this import if your file/module name differs
from model import CausalSelfAttention


def make_config(C: int, n_head: int, bias: bool, use_identity_block: bool):
    # Minimal config fields used by nanoGPT's CausalSelfAttention
    return SimpleNamespace(
        n_embd=C,
        n_head=n_head,
        bias=bias,
        dropout=0.0,  # set 0 to avoid randomness + overhead
        use_identity_block=use_identity_block,
    )


@torch.no_grad()
def bench_forward(mod: torch.nn.Module, x: torch.Tensor, warmup: int, iters: int) -> float:
    mod.eval()

    # Warmup
    for _ in range(warmup):
        _ = mod(x)

    # Time
    if x.is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = mod(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, choices=[None, "cpu", "cuda"], help="cpu/cuda (default: auto)")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--C", type=int, default=768)
    ap.add_argument("--n_head", type=int, default=12)
    ap.add_argument("--bias", action="store_true", default=False)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    args = ap.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # Keep it deterministic-ish
    torch.manual_seed(0)
    if device == "cuda":
        torch.cuda.manual_seed_all(0)

    # Instantiate baseline and pruned
    cfg_base = make_config(args.C, args.n_head, args.bias, use_identity_block=False)
    cfg_prun = make_config(args.C, args.n_head, args.bias, use_identity_block=True)

    attn_base = CausalSelfAttention(cfg_base).to(device=device)
    attn_prun = CausalSelfAttention(cfg_prun).to(device=device)

    # Match dtypes (for fair timing)
    attn_base = attn_base.to(dtype=dtype)
    attn_prun = attn_prun.to(dtype=dtype)

    # Input
    x = torch.randn(args.B, args.T, args.C, device=device, dtype=dtype)

    # Optional: enable best matmul kernels on CUDA
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    t_base = bench_forward(attn_base, x, warmup=args.warmup, iters=args.iters)
    t_prun = bench_forward(attn_prun, x, warmup=args.warmup, iters=args.iters)

    speedup = t_base / t_prun if t_prun > 0 else float("inf")

    # Report
    print(f"Device: {device} | dtype: {args.dtype}")
    print(f"Shape: B={args.B}, T={args.T}, C={args.C}, n_head={args.n_head}")
    print(f"Iters: {args.iters} (warmup {args.warmup})")
    print()
    print(f"Baseline (use_identity_block=False): {t_base*1e3:.3f} ms/iter")
    print(f"Pruned   (use_identity_block=True):  {t_prun*1e3:.3f} ms/iter")
    print(f"Speedup: {speedup:.3f}x")

    # Sanity: output shapes match
    with torch.no_grad():
        y0 = attn_base(x)
        y1 = attn_prun(x)
    print()
    print(f"Output shapes: baseline={tuple(y0.shape)}, pruned={tuple(y1.shape)}")


if __name__ == "__main__":
    print("starting")
    main()