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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", required=True)
    args = parser.parse_args()
    device = benchmark_device()

    with open(args.data_file, "rb") as f:
        train_batches = read_u64(f)
        test_batches = read_u64(f)
        batch_size = read_u64(f)
        features = read_u64(f)
        epochs = read_u64(f)
        lr = read_f32(f)
        x_train = read_f32s(f, train_batches * batch_size * features).view(train_batches, batch_size, features)
        y_train = read_f32s(f, train_batches * batch_size).view(train_batches, batch_size, 1)
        x_test = read_f32s(f, test_batches * batch_size * features).view(test_batches, batch_size, features)
        y_test = read_f32s(f, test_batches * batch_size).view(test_batches, batch_size, 1)
        w = read_f32s(f, features).view(features, 1)
        b = read_f32s(f, 1).view(1)

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    w = w.clone().to(device)
    b = b.clone().to(device)
    loss_history = []
    start = time.perf_counter()

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch in range(train_batches):
            xb = x_train[batch]
            yb = y_train[batch]
            pred = xb @ w + b.view(1, 1)
            loss = ((pred - yb) ** 2).mean()
            grad_w = (2.0 / batch_size) * xb.transpose(0, 1) @ (pred - yb)
            grad_b = (2.0 / batch_size) * (pred - yb).sum()
            w = w - lr * grad_w
            b = b - lr * grad_b.view(1)
            epoch_loss += float(loss.item())
        loss_history.append(epoch_loss / train_batches)

    eval_output = []
    for batch in range(test_batches):
        pred = x_test[batch] @ w + b.view(1, 1)
        eval_output.extend(pred.flatten().tolist())

    if device.type == "cuda":
        torch.cuda.synchronize()
    run_ms = (time.perf_counter() - start) * 1000.0

    print(
        json.dumps(
            {
                "run_ms": run_ms,
                "loss_history": loss_history,
                "final_loss": loss_history[-1],
                "final_accuracy": None,
                "output": [float(v) for v in eval_output],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
