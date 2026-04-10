from __future__ import annotations

import json
from pathlib import Path
import tempfile
import types
import urllib.request

import chelis
import safetensors.torch
import torch


LOSS_PROGRAM = """let x = (x : tensor[4, f32])
let loss = (mean(x, 0) : tensor[f32])
"""

CHECK_PROGRAM = """def relu2(x: tensor[2, f32]) -> tensor[2, f32] = relu(x)
"""


def post_json(url: str, path: str, payload: dict[str, object]) -> dict[str, object]:
    request = urllib.request.Request(
        f"{url}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    tide_url = __import__("os").environ["CHELIS_TIDE_URL"]

    check = chelis.check(CHECK_PROGRAM)
    assert isinstance(check, chelis.CheckResult)
    assert check.score == 1.0

    tide_check = post_json(
        tide_url,
        "/check",
        {"source_kind": "surf", "source": CHECK_PROGRAM},
    )
    assert tide_check["ok"] is True
    assert tide_check["result"]["score"] == check.score
    assert tide_check["result"]["typed_nodes"] == check.typed_nodes

    compile_result = chelis.compile(CHECK_PROGRAM)
    assert isinstance(compile_result, chelis.CompileResult)
    assert any(file.path.endswith(".c") for file in compile_result.files)

    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    wrapped = chelis.from_dlpack(x)
    assert wrapped.shape == (2, 3)
    roundtrip = torch.from_dlpack(wrapped)
    assert roundtrip.data_ptr() == x.data_ptr()

    class FakeGpuTensor:
        device = types.SimpleNamespace(type="cuda")

        def __dlpack__(self, stream=None):
            raise AssertionError("GPU tensors should be rejected before __dlpack__")

    try:
        chelis.from_dlpack(FakeGpuTensor())
    except ValueError as err:
        assert "3b-ii" in str(err)
    else:
        raise AssertionError("expected ValueError for GPU tensor")

    eval_input = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    eval_result = chelis.eval(LOSS_PROGRAM, {"x": eval_input})
    assert isinstance(eval_result, chelis.EvalResult)
    loss = next(root for root in eval_result.roots if root.name == "loss")
    assert loss.value.data == (2.5,)
    assert "copies Python tensor inputs" in (chelis.eval.__doc__ or "")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "weights.safetensors"
        chelis.save_safetensors(path, {"x": x})
        torch_loaded = safetensors.torch.load_file(path)
        assert torch.equal(torch_loaded["x"], x)

        chelis_loaded = chelis.load_safetensors(path)
        assert torch.equal(torch.from_dlpack(chelis_loaded["x"]), x)

        torch_path = Path(tmpdir) / "torch_weights.safetensors"
        safetensors.torch.save_file({"x": x}, torch_path)
        loaded_from_torch = chelis.load_safetensors(torch_path)
        assert torch.equal(torch.from_dlpack(loaded_from_torch["x"]), x)


if __name__ == "__main__":
    main()
