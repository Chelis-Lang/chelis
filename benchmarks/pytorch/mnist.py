#!/usr/bin/env python3
import argparse
import json
import struct
import time

import torch


def read_u64(f):
    return struct.unpack("<Q", f.read(8))[0]


def read_f32(f):
    return struct.unpack("<f", f.read(4))[0]


def read_f32s(f, count):
    return torch.tensor(struct.unpack(f"<{count}f", f.read(4 * count)), dtype=torch.float32)


def benchmark_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def relu(x):
    return torch.maximum(x, torch.zeros_like(x))


def softmax(x):
    return torch.exp(torch.log_softmax(x, dim=1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", required=True)
    args = parser.parse_args()
    device = benchmark_device()

    with open(args.data_file, "rb") as f:
        train_batches = read_u64(f)
        test_batches = read_u64(f)
        batch_size = read_u64(f)
        epochs = read_u64(f)
        lr = read_f32(f)

        x_train = [
            read_f32s(f, batch_size * 784).view(batch_size, 784)
            for _ in range(train_batches)
        ]
        y_train = [
            read_f32s(f, batch_size * 10).view(batch_size, 10)
            for _ in range(train_batches)
        ]

        x_test = [
            read_f32s(f, batch_size * 784).view(batch_size, 784)
            for _ in range(test_batches)
        ]
        y_test = [
            read_f32s(f, batch_size * 10).view(batch_size, 10)
            for _ in range(test_batches)
        ]

        w1 = read_f32s(f, 784 * 128).view(784, 128)
        b1 = read_f32s(f, 128)
        w2 = read_f32s(f, 128 * 10).view(128, 10)
        b2 = read_f32s(f, 10)

    x_train = [xb.to(device) for xb in x_train]
    y_train = [yb.to(device) for yb in y_train]
    x_test = [xb.to(device) for xb in x_test]
    y_test = [yb.to(device) for yb in y_test]
    w1 = w1.clone().to(device).requires_grad_(True)
    b1 = b1.clone().to(device).requires_grad_(True)
    w2 = w2.clone().to(device).requires_grad_(True)
    b2 = b2.clone().to(device).requires_grad_(True)

    loss_history = []
    start = time.perf_counter()
    for _ in range(epochs):
        epoch_loss = 0.0
        for xb, yb in zip(x_train, y_train):
            hidden_pre = xb @ w1 + b1
            hidden = relu(hidden_pre)
            logits = hidden @ w2 + b2
            log_probs = torch.log_softmax(logits, dim=1)
            loss = -(log_probs * yb).sum(dim=1).mean()
            loss.backward()

            with torch.no_grad():
                w1 -= lr * w1.grad
                b1 -= lr * b1.grad
                w2 -= lr * w2.grad
                b2 -= lr * b2.grad
                w1.grad.zero_()
                b1.grad.zero_()
                w2.grad.zero_()
                b2.grad.zero_()
            epoch_loss += float(loss.item())
        loss_history.append(epoch_loss / train_batches)

    eval_output = []
    correct = 0
    for xb, yb in zip(x_test, y_test):
        logits = relu(xb @ w1 + b1) @ w2 + b2
        eval_output.extend(logits.flatten().cpu().tolist())
        correct += int((logits.argmax(dim=1) == yb.argmax(dim=1)).sum().item())

    if device.type == "cuda":
        torch.cuda.synchronize()

    print(
        json.dumps(
            {
                "run_ms": (time.perf_counter() - start) * 1000.0,
                "loss_history": loss_history,
                "final_loss": loss_history[-1],
                "final_accuracy": correct / (test_batches * batch_size),
                "output": eval_output,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
