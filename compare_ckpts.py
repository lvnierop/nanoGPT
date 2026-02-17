#!/usr/bin/env python3
"""
compare_ckpts.py

End-to-end comparison for two nanoGPT checkpoints.

Usage (run from the nanoGPT repo root so `model.py` is importable):
  python compare_ckpts.py /path/to/ckpt_dir_A /path/to/ckpt_dir_B --device cpu
  python compare_ckpts.py ckptA ckptB --device cuda --dtype float16 --trials 5 --seq 64
"""

import os
import sys
import math
import argparse
import torch

# nanoGPT model definitions (must be available on PYTHONPATH; easiest: run from repo root)
from model import GPT, GPTConfig



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_a", type=str)
    ap.add_argument("dir_b", type=str)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--tol", type=float, default=1e-5, help="max|diff| threshold for 'same for our purposes'")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()


if __name__ == "__main__":
    main()
