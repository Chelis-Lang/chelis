# Core-fragment Linux release receipt — 2026-10-03

Owner: [#2102](https://github.com/Chelis-Lang/chelis/issues/2102), part of the [launch ledger](https://github.com/Chelis-Lang/chelis/issues/1362).

## Pinned inputs

- Compiler source: `c39ff41ed2d2fb3ecc3d582bc3e3afa71ee89d36` (main after #3028); version `chelis 0.18.12`, sealed runtime.
- Compiler SHA256: `dac9d989a3b40763b77934b46732167b7849e5d2cf204d90229c254789bd1fad`.
- Runtime archive SHA256: `33d02f20aea8c7a460e2404e0bb348efaa0ae41a5434009aa64b04706d502dc4`.
- Platform: Linux x86_64. Compile profile: `emitted` (the compiler’s printed command, including `-ffp-contract=off -fno-fast-math`).
- Manifest: version 3; 13 required value cases, 119 exclusions, 132 discovered sources. No case, root, source revision or exclusion changed.
- C Note: `960a9beb458fc794086a2a36ecc93c8979ece1d0`.
- Sonar: `9b26133f66988ef6643ff0d2fdab1ce90d104714`.

| Header | SHA256 |
|---|---|
| `chelis_blas.h` | `efebbfefde80c50386058f11623acdbf0a77e8a0a098619adc198280f8c17203` |
| `chelis_math.h` | `47255b844a128e88086b475591d48374b6e87fe63ed796b21c7c1d98faf90508` |
| `chelis_runtime.h` | `18de8f2dada522bd073c121af925e1e880c7245e03fc7fbdd6f9d09e9764279d` |
| `chelis_runtime_dtype.h` | `942c1330ea777f75729f2a2d3e234975711634fec97e38a0cf2780c17b6d4b65` |
| `chelis_runtime_views.h` | `e4317e87e1b1514a89fd0042ffd29a6b3622b1ff0c9fcce76e4bcd3bc0941477` |
| `chelis_simd.h` | `c6b6b33ce45b79306021069fabbe3f69ca4890f27f769c84db20ae197bbd3eb4` |

## Required case results

Initial measurement: 11 `agree` and two `unexpected-pass` verdicts; all 13 observed streams agreed. The two now-passing #2379 rows are promoted to ordinary required cases by removing only their obsolete divergence metadata. The promoted-manifest run exits 0 with `RECEIPT: PASS`: 13 `agree`, zero divergences, zero pin failures and zero non-vacuity failures. The 148 supporting unit tests also pass.

| Case | Eval | Compiled stage | Agreement | Truncated root |
|---|---|---|---|---|
| `c-note/docs/qa_evidence/ground_truth/bs_pricing.ch` | exit 0 | run, exit 0 | exact | no |
| `c-note/docs/qa_evidence/ground_truth/greeks.ch` | exit 0 | run, exit 0 | exact | no |
| `c-note/docs/qa_evidence/ground_truth/square.ch` | exit 0 | run, exit 0 | exact | no |
| `c-note/docs/qa_evidence/ground_truth/upper.ch` | exit 0 | run, exit 0 | exact | no |
| `c-note/fixtures/black_scholes/call.ch` | exit 0 | run, exit 0 | exact | no |
| `c-note/fixtures/exemplars/black_scholes/source.ch` | exit 0 | run, exit 0 | exact | no |
| `c-note/fixtures/templates/sources/black_scholes_greeks.ch` | exit 0 | run, exit 0 | exact | no |
| `c-note/fixtures/templates/sources/two_period_bond.ch` | exit 0 | run, exit 0 | exact | no |
| `c-note/fixtures/tensor/grad_output.ch` | exit 0 | run, exit 0 | exact | no |
| `c-note/fixtures/tensor/vector_output.ch` | exit 0 | run, exit 0 | exact | no |
| `c-note/src/lib.ch` | exit 0 | run, exit 0 | exact | no |
| `sonar/landing/fixtures/builtin_parameter.ch` | exit 0 | run, exit 0 | exact | no |
| `sonar/landing/review/economoist-release/tensor-source.ch` | exit 0 | run, exit 0 | exact | no |

## Acceptance oracle

Use task-owned paths for the sealed compiler and pinned corpus checkouts. Build the compiler into a dedicated target that subsequent default-feature commands do not overwrite. With C Note and Sonar checked out at the revisions above:

```text
cargo build -p chelis-cli --features sealed-runtime --target-dir target/receipt-after-math-build
.venv/bin/python scripts/core_fragment_parity_receipt.py --chelis target/receipt-after-math-build/debug/chelis --corpus c-note=/path/to/pinned/c-note --corpus sonar=/path/to/pinned/sonar --out target/release-receipt
```

Acceptance is exit 0 and `RECEIPT: PASS`, with all 13 required rows agreeing and no pin or non-vacuity failure. The runner records actual commands, compiler/archive/header pins, case verdicts and observation coverage in `receipt.json`. Unit tests exercise lane splits, missing roots, bad pins, compile failures and refusal to accept tracked failures; they support this oracle.

## Scope

This is evidence for the recorded Linux compiler, toolchain, flags and 13 pinned value cases. It does not prove every program or platform, a real trap case, or the wider #2379 acceptance surface. That issue remains with its owner. Voyage capture/provenance is nonblocking #3051, assigned to Makis (`glampouras`). No unavailable capture has been counted as tested.
