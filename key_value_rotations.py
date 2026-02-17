import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import copy


out_dir = 'out-shakespeare-char'
gauged_dir = 'out-shakespeare-char-gauged'
random_dir = 'out-shakespeare-char-random'
gauge_fixed_dir = 'out-shakespeare-char-gauge-fixed'
device = 'cpu'
dtype = 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file

device_type = 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# ============================================================================
# STEP 1: Load original model and preserve checkpoint metadata
# ============================================================================
# Load model from checkpoint
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
ckpt_path_gauged = os.path.join(gauged_dir, 'ckpt.pt')
ckpt_path_random = os.path.join(random_dir, 'ckpt.pt')
ckpt_path_gauge_fixed = os.path.join(gauge_fixed_dir, 'ckpt.pt')
os.makedirs(gauged_dir, exist_ok=True)
os.makedirs(random_dir, exist_ok=True)
os.makedirs(gauge_fixed_dir, exist_ok=True)

original_checkpoint = torch.load(ckpt_path, map_location=device)
gauged_checkpoint = copy.deepcopy(original_checkpoint)
random_checkpoint = copy.deepcopy(original_checkpoint)
gauge_fixed_checkpoint = copy.deepcopy(original_checkpoint)
gptconf = GPTConfig(**original_checkpoint['model_args'])
original_model = GPT(gptconf)
original_state_dict = original_checkpoint['model'].copy()
unwanted_prefix = '_orig_mod.'
for k,v in list(original_state_dict.items()):
    if k.startswith(unwanted_prefix):
        original_state_dict[k[len(unwanted_prefix):]] = original_state_dict.pop(k)
original_model.load_state_dict(original_state_dict)
original_model.eval()
original_model.to(device)

cfg = original_checkpoint["config"]

n_layer = cfg["n_layer"]   # number of transformer blocks
n_head  = cfg["n_head"]    # number of attention heads
n_embd  = cfg["n_embd"]    # embedding size

sd = gauged_checkpoint["model"]
d_head = n_embd // n_head

for i in range(n_layer):
    W = sd[f"transformer.h.{i}.attn.c_attn.weight"]   # (3*n_embd, n_embd)
    Wq = W[:n_embd, :]
    Wk = W[n_embd:2*n_embd, :]

    for h in range(n_head):
        sl = slice(h*d_head, (h+1)*d_head)

        A = torch.randn(d_head, d_head, device=W.device, dtype=W.dtype)
        Ainv = torch.linalg.inv(A)

        Wq_h = Wq[sl, :].clone()
        Wk_h = Wk[sl, :].clone()

        # implements q -> q A and k -> k (A^{-1})^T (logits cancel)
        Wq[sl, :] = A.T @ Wq_h
        Wk[sl, :] = Ainv @ Wk_h


# copy checpoint for modification
torch.save(gauged_checkpoint, ckpt_path_gauged)

sd = random_checkpoint["model"]
d_head = n_embd // n_head

for i in range(n_layer):
    W = sd[f"transformer.h.{i}.attn.c_attn.weight"]   # (3*n_embd, n_embd)
    Wq = W[:n_embd, :]
    Wk = W[n_embd:2*n_embd, :]

    for h in range(n_head):
        sl = slice(h*d_head, (h+1)*d_head)

        A = torch.randn(d_head, d_head, device=W.device, dtype=W.dtype)
        Ainv = torch.linalg.inv(A)

        Wq_h = Wq[sl, :].clone()
        Wk_h = Wk[sl, :].clone()

        # implements q -> q A and k -> k (A^{-1})^T (logits cancel)
        Wq[sl, :] = A.T @ Wq_h
        Wk[sl, :] = Ainv @ Wk_h

torch.save(random_checkpoint, ckpt_path_random)


sd = gauge_fixed_checkpoint["model"]
d_head = n_embd // n_head

for i in range(n_layer):
    W = sd[f"transformer.h.{i}.attn.c_attn.weight"]   # (3*n_embd, n_embd)
    Wq = W[:n_embd, :]
    Wk = W[n_embd:2*n_embd, :]
    print(f"\nLayer {i} Wq BEFORE:")
    print(Wq)
    print(f"\nLayer {i} Wk BEFORE:")
    print(Wk)
    print(f"shape of Wq: {Wq.shape}")
    print(f"shape of Wk: {Wk.shape}")

    for h in range(n_head):
        sl = slice(h*d_head, (h+1)*d_head)

        Wq_h = Wq[sl, :].clone()
        Wk_h = Wk[sl, :].clone()

        B = Wq_h[:, :d_head]                 # (d, d)
        A_T = torch.linalg.inv(B)       # this is A^T
        Ainv = B.T

        # implements q -> q A and k -> k (A^{-1})^T (logits cancel)
        Wq[sl, :] = A_T @ Wq_h
        Wk[sl, :] = Ainv @ Wk_h
    print(f"\nLayer {i} Wq AFTER:")
    print(Wq)
    print(f"\nLayer {i} Wk AFTER:")
    print(Wk)
    print(Wq[:, :d_head].round(decimals=3))
torch.save(gauge_fixed_checkpoint, ckpt_path_gauge_fixed)

# Layer norm 1: state_dict[f'transformer.h.{i}.ln_1.weight']  # shape: (n_embd,)


# For each transformer block i (0 to n_layer-1):
# - Layer norm 1: state_dict[f'transformer.h.{i}.ln_1.weight']  # shape: (n_embd,)
#                 state_dict[f'transformer.h.{i}.ln_1.bias']  # shape: (n_embd,) if bias=True
# - Attention QKV projection: state_dict[f'transformer.h.{i}.attn.c_attn.weight']  # shape: (n_embd, 3*n_embd)
#                            state_dict[f'transformer.h.{i}.attn.c_attn.bias']  # shape: (3*n_embd,) if bias=True
# - Attention output projection: state_dict[f'transformer.h.{i}.attn.c_proj.weight']  # shape: (n_embd, n_embd)
#                                state_dict[f'transformer.h.{i}.attn.c_proj.bias']  # shape: (n_embd,) if bias=True
# - Layer norm 2: state_dict[f'transformer.h.{i}.ln_2.weight']  # shape: (n_embd,)
#                 state_dict[f'transformer.h.{i}.ln_2.bias']  # shape: (n_embd,) if bias=True
# - MLP first layer: state_dict[f'transformer.h.{i}.mlp.c_fc.weight']  # shape: (n_embd, 4*n_embd)
#                    state_dict[f'transformer.h.{i}.mlp.c_fc.bias']  # shape: (4*n_embd,) if bias=True
# - MLP second layer: state_dict[f'transformer.h.{i}.mlp.c_proj.weight']  # shape: (4*n_embd, n_embd)
#                     state_dict[f'transformer.h.{i}.mlp.c_proj.bias']  # shape: (n_embd,) if bias=True

# Example: Access weights directly from the loaded model
# You can also access weights via loaded_model.state_dict() or directly via model attributes:
# - loaded_model.transformer.wte.weight  # token embeddings
# - loaded_model.transformer.h[0].attn.c_attn.weight  # QKV weights for first block
# - loaded_model.transformer.h[0].attn.c_proj.weight  # attention output weights for first block
# - loaded_model.transformer.h[0].mlp.c_fc.weight  # MLP first layer weights for first block
# - loaded_model.transformer.h[0].mlp.c_proj.weight  # MLP second layer weights for first block