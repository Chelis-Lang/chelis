#!/usr/bin/env python3

import json
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LINREG = ROOT / "examples" / "linreg.ch"
MNIST = ROOT / "examples" / "mnist.ch"

LOSS_PROGRAM = """let x = (x : tensor[4, f32])
let loss = (mean(x, 0) : tensor[f32])
"""

BROKEN_PROGRAM = "let bad = 1 + true\n"
FIXED_PROGRAM = """let x = (x : tensor[4, f32])
let out = (add(x, x) : tensor[4, f32])
"""
MUL_PROGRAM = """let a = (a : tensor[2, 3, f32])
let b = (b : tensor[3, 4, f32])
let out = (matmul(a, b) : tensor[2, 4, f32])
"""
GRAD_PROGRAM = LOSS_PROGRAM


class CheckFailure(RuntimeError):
    pass


def main() -> int:
    failures = []
    try:
        http_results = run_http_suite()
        mcp_results = run_mcp_suite()
        repair_results = run_mcp_repair_workflow()
        payload_results = run_large_payload_suite()
        shutdown_results = run_shutdown_suite()
    except CheckFailure as err:
        print(f"fatal: {err}", file=sys.stderr)
        return 1

    print_section("HTTP Server", http_results, failures)
    print_section("MCP Server", mcp_results, failures)
    print_section("Agent-Repair Workflow", repair_results, failures)
    print_section("Larger Payloads", payload_results, failures)
    print_section("Shutdown", shutdown_results, failures)

    if failures:
        print(f"\nFAILED: {len(failures)} checks failed", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("\nPASS: all Phase 2e external checks succeeded")
    return 0


def print_section(title, rows, failures):
    print(f"\n{title}")
    for name, ok, detail in rows:
        status = "PASS" if ok else "FAIL"
        print(f"- {name}: {status} — {detail}")
        if not ok:
            failures.append(f"{title}: {name}: {detail}")


def run_http_suite():
    port = free_port()
    proc = spawn(["cargo", "run", "-q", "-p", "chelis-cli", "--", "tide", "serve", "--port", str(port)])
    try:
        wait_for_http(port)
        rows = []

        payload = {"source_kind": "surf", "source": LOSS_PROGRAM}
        body = http_json(port, "/check", payload)
        rows.append(("check happy path", body["ok"] and body["result"]["score"] == 1.0, f"score={body['result']['score']}"))

        payload = {
            "source_kind": "surf",
            "source": LOSS_PROGRAM,
            "bindings": {"x": {"shape": [4], "data": [1.0, 2.0, 3.0, 4.0]}},
        }
        body = http_json(port, "/eval", payload)
        loss_root = next(root for root in body["result"]["roots"] if root["name"] == "loss")
        rows.append(("eval named bindings", body["ok"] and loss_root["value"]["data"] == [2.5], f"loss={loss_root['value']['data']}"))

        body = http_json(port, "/eval", {"source_kind": "surf", "source": LOSS_PROGRAM, "bindings": {}})
        rows.append(("eval missing binding", (not body["ok"]) and body["stage"] == "eval", f"stage={body.get('stage')}"))

        payload = {
            "source_kind": "surf",
            "source": MUL_PROGRAM,
            "bindings": {
                "b": {"shape": [3, 4], "data": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]},
                "a": {"shape": [2, 3], "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},
            },
        }
        body = http_json(port, "/eval", payload)
        out_root = next(root for root in body["result"]["roots"] if root["name"] == "out")
        rows.append((
            "eval reversed binding order",
            body["ok"] and out_root["value"]["shape"] == [2, 4] and out_root["value"]["data"] == [1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 0.0],
            f"shape={out_root['value']['shape']}",
        ))

        payload = {
            "source_kind": "surf",
            "source": GRAD_PROGRAM,
            "output_name": "loss",
            "wrt_names": ["x"],
        }
        body = http_json(port, "/grad", payload)
        rows.append((
            "grad",
            body["ok"] and "x" in body["result"]["grad_nodes_by_name"] and "loss" in body["result"]["forward_nodes_by_name"],
            "maps present" if body["ok"] else body,
        ))

        return rows
    finally:
        stop_process(proc, signal.SIGINT)


def run_mcp_suite():
    init, tools = mcp_roundtrip([
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    ])
    tool_names = [tool["name"] for tool in tools["result"]["tools"]]
    rows = [
        ("tools/list", len(tool_names) == 7 and tool_names == [
            "chelis_check",
            "chelis_compile",
            "chelis_desugar",
            "chelis_decompile",
            "chelis_eval",
            "chelis_grad",
            "chelis_validate",
        ], f"tools={tool_names}")
    ]

    rows.append(run_mcp_tool("chelis_check", {"source_kind": "surf", "source": LOSS_PROGRAM}, lambda payload: payload["ok"] and payload["result"]["score"] == 1.0, "score"))
    rows.append(run_mcp_tool("chelis_eval", {
        "source_kind": "surf",
        "source": LOSS_PROGRAM,
        "bindings": {"x": {"shape": [4], "data": [1.0, 2.0, 3.0, 4.0]}},
    }, lambda payload: payload["ok"] and next(root for root in payload["result"]["roots"] if root["name"] == "loss")["value"]["data"] == [2.5], "loss"))
    rows.append(run_mcp_tool("chelis_grad", {
        "source_kind": "surf",
        "source": GRAD_PROGRAM,
        "output_name": "loss",
        "wrt_names": ["x"],
    }, lambda payload: payload["ok"] and "x" in payload["result"]["grad_nodes_by_name"], "grad map"))
    return rows


def run_mcp_repair_workflow():
    responses = mcp_roundtrip([
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "chelis_check", "arguments": {"source_kind": "surf", "source": BROKEN_PROGRAM}}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "chelis_check", "arguments": {"source_kind": "surf", "source": FIXED_PROGRAM}}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "chelis_compile", "arguments": {"source_kind": "surf", "source": FIXED_PROGRAM, "target": "c"}}},
    ])
    broken = structured_content(responses[1])
    fixed = structured_content(responses[2])
    compiled = structured_content(responses[3])
    return [
        ("broken snippet", broken["ok"] and broken["result"]["score"] < 1.0 and len(broken["result"]["errors"]) > 0, f"score={broken['result']['score']}"),
        ("fixed snippet", fixed["ok"] and fixed["result"]["score"] == 1.0, f"score={fixed['result']['score']}"),
        (
            "compile",
            compiled["ok"] and len(compiled["result"]["files"]) == 4,
            f"files={len(compiled['result']['files'])}" if compiled["ok"] else f"stage={compiled.get('stage')}",
        ),
    ]


def run_large_payload_suite():
    port = free_port()
    proc = spawn(["cargo", "run", "-q", "-p", "chelis-cli", "--", "tide", "serve", "--port", str(port)])
    try:
        wait_for_http(port)
        rows = []
        linreg = LINREG.read_text()
        mnist = MNIST.read_text()

        desugar = http_json(port, "/desugar", {"source": linreg})
        rows.append(("linreg /desugar", desugar["ok"] and len(desugar["result"]["deep_text"]) > 0, f"chars={len(desugar['result']['deep_text'])}"))

        check = http_json(port, "/check", {"source_kind": "surf", "source": linreg})
        rows.append(("linreg /check", check["ok"] and check["result"]["score"] == 1.0, f"score={check['result']['score']}"))

        compile_result = http_json(port, "/compile", {"source_kind": "surf", "source": linreg, "target": "c"})
        rows.append(("linreg /compile", compile_result["ok"] and len(compile_result["result"]["files"]) == 4, f"files={len(compile_result['result']['files'])}"))

        check = http_json(port, "/check", {"source_kind": "surf", "source": mnist})
        rows.append(("mnist /check", check["ok"] and check["result"]["score"] == 1.0, f"score={check['result']['score']}"))

        start = time.time()
        compile_result = http_json(port, "/compile", {"source_kind": "surf", "source": mnist, "target": "c"})
        elapsed_ms = int((time.time() - start) * 1000)
        rows.append(("mnist /compile", compile_result["ok"] and len(compile_result["result"]["files"]) == 4, f"files={len(compile_result['result']['files'])}"))
        rows.append(("mnist latency", compile_result["ok"], f"{elapsed_ms}ms compile time"))
        return rows
    finally:
        stop_process(proc, signal.SIGINT)


def run_shutdown_suite():
    rows = []

    port = free_port()
    proc = spawn(["cargo", "run", "-q", "-p", "chelis-cli", "--", "tide", "serve", "--port", str(port)])
    wait_for_http(port)
    code = stop_process(proc, signal.SIGINT)
    rows.append(("SIGINT", code == 130 or code == -signal.SIGINT, f"exit={code}"))

    port = free_port()
    proc = spawn(["cargo", "run", "-q", "-p", "chelis-cli", "--", "tide", "serve", "--port", str(port)])
    wait_for_http(port)
    code = stop_process(proc, signal.SIGTERM)
    rows.append(("SIGTERM", code == 143 or code == -signal.SIGTERM, f"exit={code}"))

    proc = spawn(["cargo", "run", "-q", "-p", "chelis-cli", "--", "tide", "mcp"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.stdin.close()
    code = proc.wait(timeout=5)
    rows.append(("MCP EOF", code == 0, f"exit={code}"))

    responses = mcp_roundtrip([
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "shutdown", "params": {}},
    ], send_exit=True)
    rows.append(("MCP shutdown+exit", len(responses) == 2, f"responses={len(responses)}"))
    return rows


def run_mcp_tool(name, arguments, predicate, detail_label):
    (_, response) = mcp_roundtrip([
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": name, "arguments": arguments}},
    ])
    payload = structured_content(response)
    ok = predicate(payload)
    detail = payload["result"]["score"] if detail_label == "score" and payload["ok"] else detail_label
    if detail_label == "loss" and payload["ok"]:
        detail = next(root for root in payload["result"]["roots"] if root["name"] == "loss")["value"]["data"]
    return (name, ok, f"{detail_label}={detail}")


def http_json(port, path, payload):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


def wait_for_http(port):
    deadline = time.time() + 20
    last_error = None
    while time.time() < deadline:
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/check",
                data=json.dumps({"source_kind": "surf", "source": LOSS_PROGRAM}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=2):
                return
        except Exception as err:  # noqa: BLE001
            last_error = err
            time.sleep(0.25)
    raise CheckFailure(f"HTTP server failed to start on port {port}: {last_error}")


def mcp_roundtrip(messages, send_exit=False):
    payload = b"".join(encode_mcp(message) for message in messages)
    if send_exit:
        payload += encode_mcp({"jsonrpc": "2.0", "method": "exit", "params": {}})
    proc = subprocess.run(
        ["cargo", "run", "-q", "-p", "chelis-cli", "--", "tide", "mcp"],
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=ROOT,
        timeout=20,
    )
    if proc.returncode != 0:
        raise CheckFailure(f"MCP process failed: {proc.stderr.decode()}")
    return decode_mcp_stream(proc.stdout)


def structured_content(response):
    return response["result"]["structuredContent"]


def encode_mcp(message):
    body = json.dumps(message)
    return f"Content-Length: {len(body)}\r\n\r\n{body}".encode()


def decode_mcp_stream(raw):
    offset = 0
    messages = []
    while offset < len(raw):
        header_end = raw.find(b"\r\n\r\n", offset)
        if header_end == -1:
            break
        header = raw[offset:header_end].decode()
        length = None
        for line in header.split("\r\n"):
            if line.lower().startswith("content-length:"):
                length = int(line.split(":", 1)[1].strip())
        if length is None:
            raise CheckFailure("MCP response missing Content-Length")
        body_start = header_end + 4
        body_end = body_start + length
        messages.append(json.loads(raw[body_start:body_end].decode()))
        offset = body_end
    return messages


def spawn(argv, **kwargs):
    return subprocess.Popen(argv, cwd=ROOT, **kwargs)


def stop_process(proc, sig):
    if proc.poll() is not None:
        return proc.returncode
    proc.send_signal(sig)
    try:
        return proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        return proc.wait(timeout=5)


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


if __name__ == "__main__":
    sys.exit(main())
