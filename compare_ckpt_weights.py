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

def load_model(
        path,
        device="cpu"
):
    ckpt_path = os.path.join(path, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconfig = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconfig)
    state_dict = checkpoint["model"].copy()
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    try:
        print(f"loaded model from {path}, Passthrough: {model.config.use_identity_block}")
    except Exception as e:
        print(e)
        
    return model



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_a", type=str)
    ap.add_argument("dir_b", type=str)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--tol", type=float, default=1e-5, help="max|diff| threshold for 'same for our purposes'")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    model_a = load_model(args.dir_a, device=args.device)
    model_b = load_model(args.dir_b, device=args.device)

    n_layers = model_a.config.n_layer
    n_heads = model_a.config.n_head
    n_embd = model_a.config.n_embd

    block_size = model_a.config.block_size
    vocab_size = model_a.config.vocab_size

    for layer in range(n_layers):
        Wqa = model_a.state_dict()[f"transformer.h.{layer}.attn.c_attn.weight"][:n_embd, :]
        Wka = model_a.state_dict()[f"transformer.h.{layer}.attn.c_attn.weight"][n_embd:2*n_embd, :]
        Wva = model_a.state_dict()[f"transformer.h.{layer}.attn.c_attn.weight"][2*n_embd:3*n_embd, :]

        Wqb = model_b.state_dict()[f"transformer.h.{layer}.attn.c_attn.weight"][:n_embd, :]
        Wkb = model_b.state_dict()[f"transformer.h.{layer}.attn.c_attn.weight"][n_embd:2*n_embd, :]
        Wvb = model_b.state_dict()[f"transformer.h.{layer}.attn.c_attn.weight"][2*n_embd:3*n_embd, :]

        diffs = torch.abs(Wqa - Wqb)
        print(f"Wq, layer: {layer}, max diff: {torch.max(diffs)}, mean diff: {torch.mean(diffs)}")

        diffs = torch.abs(Wka - Wkb)
        print(f"Wk, layer: {layer}, max diff: {torch.max(diffs)}, mean diff: {torch.mean(diffs)}")

        diffs = torch.abs(Wva - Wvb)
        print(f"Wv, layer: {layer}, max diff: {torch.max(diffs)}, mean diff: {torch.mean(diffs)}")







if __name__ == "__main__":
    main()
