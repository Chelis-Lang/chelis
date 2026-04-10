from __future__ import annotations

from pathlib import Path
import tempfile
import warnings

import chelis
import numpy as np
import torch


PROGRAM = """def relu4(x: tensor[4, f32]) -> tensor[4, f32] = relu(x)
"""


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source_path = root / "model.ch"
        source_path.write_text(PROGRAM)

        compiled = chelis.compile_and_load(
            source_path,
            artifact_dir=root / "artifacts",
        )
        assert isinstance(compiled, chelis.CompiledModel)
        assert compiled.target == "c"
        assert compiled.path.endswith("model.so")
        assert compiled.input_names == ("x",)
        assert compiled.output_names == ("root0",)

        x_torch = torch.tensor([-1.0, 2.0, -3.0, 4.0], dtype=torch.float32)
        out_torch = torch.from_dlpack(compiled(x=x_torch))
        assert torch.equal(out_torch, torch.tensor([0.0, 2.0, 0.0, 4.0]))

        x_numpy = np.arange(4, dtype=np.float32)
        compiled_numpy = compiled(x=x_numpy)
        assert np.array_equal(np.from_dlpack(compiled_numpy), np.arange(4, dtype=np.float32))

        base = np.arange(8, dtype=np.float32).reshape(2, 4)
        strided = base[1]
        wrapped = chelis.from_dlpack(strided)
        roundtrip = np.from_dlpack(wrapped)
        assert roundtrip.__array_interface__["data"][0] == strided.__array_interface__["data"][0]
        assert np.array_equal(roundtrip, strided)

        try:
            compiled(x=torch.ones(4, dtype=torch.float64))
        except ValueError as err:
            assert "float32" in str(err)
        else:
            raise AssertionError("expected ValueError for unsupported dtype")

        try:
            compiled(x=torch.ones(5, dtype=torch.float32))
        except ValueError as err:
            assert "axis 0 expected 4" in str(err)
        else:
            raise AssertionError("expected ValueError for shape mismatch")

        if torch.version.hip and torch.cuda.is_available():
            hip_compiled = chelis.compile_and_load(
                source_path,
                target="hip",
                artifact_dir=root / "artifacts_hip",
            )
            x_gpu = x_torch.to("cuda")
            hip_out = torch.from_dlpack(hip_compiled(x=x_gpu)).cpu()
            assert torch.equal(hip_out, torch.tensor([0.0, 2.0, 0.0, 4.0]))

        source_path.write_text(
            "def relu4(x: tensor[4, f32]) -> tensor[4, f32] = neg(x)\n"
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loaded = chelis.load(compiled.path)
            assert isinstance(loaded, chelis.CompiledModel)
            assert any("source has changed since this artifact was compiled" in str(w.message) for w in caught)
            loaded_out = torch.from_dlpack(loaded(x=x_torch))
            assert torch.equal(loaded_out, torch.tensor([0.0, 2.0, 0.0, 4.0]))


if __name__ == "__main__":
    main()
