#!/usr/bin/env python3
import argparse
import json
import struct
import time

import torch


def read_u64(f):
    return struct.unpack("<Q", f.read(8))[0]


def read_f32s(f, count):
    return torch.tensor(struct.unpack(f"<{count}f", f.read(4 * count)), dtype=torch.float32)


def benchmark_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim=1, keepdim=True)
    centered = x - mean
    var = (centered * centered).mean(dim=1, keepdim=True)
    inv = torch.rsqrt(var + eps)
    return centered * inv * gamma.view(1, -1) + beta.view(1, -1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--iterations", type=int, default=None)
    args = parser.parse_args()
    device = benchmark_device()

    with open(args.data_file, "rb") as f:
        seq_len = read_u64(f)
        d_model = read_u64(f)
        n_heads = read_u64(f)
        head_dim = read_u64(f)
        d_ff = read_u64(f)
        file_iters = read_u64(f)
        x = read_f32s(f, seq_len * d_model).view(seq_len, d_model)
        heads = []
        for _ in range(n_heads):
            heads.append(
                {
                    "wq": read_f32s(f, d_model * head_dim).view(d_model, head_dim),
                    "wk": read_f32s(f, d_model * head_dim).view(d_model, head_dim),
                    "wv": read_f32s(f, d_model * head_dim).view(d_model, head_dim),
                    "wo": read_f32s(f, head_dim * d_model).view(head_dim, d_model),
                }
            )
        ff1 = read_f32s(f, d_model * d_ff).view(d_model, d_ff)
        ff2 = read_f32s(f, d_ff * d_model).view(d_ff, d_model)
        gamma1 = read_f32s(f, d_model)
        beta1 = read_f32s(f, d_model)
        gamma2 = read_f32s(f, d_model)
        beta2 = read_f32s(f, d_model)

    x = x.to(device)
    for head in heads:
        for key, value in head.items():
            head[key] = value.to(device)
    ff1 = ff1.to(device)
    ff2 = ff2.to(device)
    gamma1 = gamma1.to(device)
    beta1 = beta1.to(device)
    gamma2 = gamma2.to(device)
    beta2 = beta2.to(device)

    iters = args.iterations or file_iters
    out = None
    start = time.perf_counter()
    for _ in range(iters):
        attn_acc = None
        for head in heads:
            q = x @ head["wq"]
            k = x @ head["wk"]
            v = x @ head["wv"]
            probs = torch.softmax(q @ k.transpose(0, 1), dim=1)
            ctx = probs @ v
            proj = ctx @ head["wo"]
            attn_acc = proj if attn_acc is None else attn_acc + proj
        norm1 = layer_norm(x + attn_acc, gamma1, beta1)
        ff_hidden_pre = norm1 @ ff1
        ff_hidden = torch.maximum(ff_hidden_pre, torch.zeros_like(ff_hidden_pre))
        out = layer_norm(norm1 + ff_hidden @ ff2, gamma2, beta2)

    if device.type == "cuda":
        torch.cuda.synchronize()

    print(
        json.dumps(
            {
                "run_ms": (time.perf_counter() - start) * 1000.0,
                "loss_history": [],
                "final_loss": None,
                "final_accuracy": None,
                "output": out.flatten().cpu().tolist(),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
