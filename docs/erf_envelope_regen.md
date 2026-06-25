# erf Envelope: Sollya + Gappa Proof, Arb Cross-check, Regeneration (WI-13)

The `chelis-prove` crate ships a committed, certified `erf` envelope —
`crates/chelis-prove/data/erf_envelope.json` — that the runtime discharge and
Beacon's `erf` relaxation (WI-B6) consume to bound `erf` soundly with **no
runtime dependency** on Sollya, Gappa, or Arb. This document covers what is
committed, how it is proved and cross-checked, how CI re-validates it, and the
environment each step needs.

## What the committed data is

A piecewise envelope of contiguous boxes covering `[-300, 300]`:

| Box          | Arm                          | `eps`     | `proof_kind` |
|--------------|------------------------------|-----------|--------------|
| `[-300, -3]` | saturation `-1`              | `~2.2e-5` | `gappa`      |
| `[-3, 3]`    | central degree-21 polynomial | `~5.7e-7` | `gappa`      |
| `[3, 300]`   | saturation `+1`              | `~2.2e-5` | `gappa`      |

Each box's `eps` is a sound sup-norm bound `sup_{x in box} |approx(x) - erf(x)|
<= eps`, and **every arm carries a machine-checkable Gappa proof term**
(`proof_kind = gappa`). The committed bundle under
`crates/chelis-prove/data/erf_proof/` is a set of Gappa scripts an auditor
re-runs through `gappa` to confirm each bound:

- **Central arm.** `central_<k>.gappa` — the degree-21 remez polynomial's
  approximation error, bounded per sub-interval. The coefficients are rounded to
  `f64` **before** the proof, so the proof is about the exact polynomial the
  runtime uses.
- **Saturation tails.** `tail_pos_<k>.gappa` / `tail_neg_<k>.gappa` — the same
  machinery with the constant `+-1` as the approximation. Gappa proves
  `|+-1 - T(x)| <= bound` where `T` is the certified local Taylor model of `erf`;
  this is pure polynomial arithmetic (Gappa never sees `erf`/`exp`, which it
  cannot model). Each tail is covered by a fine "active zone" near the inner edge
  (where `|+-1 - erf|` is largest, `~2.2e-5` at `x = 3`) plus wider far
  sub-intervals out to `+-300`, all proving the bound stays under the committed
  tail `eps`.

`ProofKind::ArbEnclosure` is the honest fallback for any future arm a tool
genuinely cannot prove; no committed arm uses it today.

The committed envelope evaluates in pure `f64`. A deployed release binary gains
no FLINT/Arb/Sollya/Gappa link (`arb` is off by default and absent from the
`default` and `smt` dependency trees; Sollya and Gappa are offline tools).

## Trust model: proof term + independent cross-check

- **Central proof.** Sollya computes the remez polynomial and a *certified local
  Taylor model* of `erf` per sub-interval; Gappa machine-checks
  `|p(x) - T(x)| <= bound`, which with the certified Taylor remainder gives
  `|p - erf| <= eps`. The polynomial is evaluated in a per-sub-interval centered
  variable so the high-degree difference stays well conditioned and Gappa proves
  each sub-interval in well under a second.
- **Arb cross-check (belt + suspenders).** The WI-14 Arb whole-box certifier
  re-validates **every** box's committed `eps` on every CI build — both the
  Gappa central arm and the Arb tails — by asserting the committed `eps` bounds
  the Arb sup-norm bound. The central arm thus has both a Gappa proof term and an
  independent Arb confirmation.
- **Consistency.** Tests (Rust and Python) lock the committed central polynomial
  and `eps` to the Gappa proof bundle's manifest, so the proved bound and the
  consumed bound cannot drift apart, and the envelope provenance pins the
  bundle's sha256.

## Regeneration pipeline

```sh
# 0. one-time: build the Sollya + Gappa toolchain (see Environment below)
.venv/bin/python scripts/setup_proof_toolchain.py

# 1. central Gappa proof bundle (Sollya remez + per-sub-interval Gappa proofs)
.venv/bin/python scripts/generate_erf_proof.py

# 2. assemble the committed envelope: Gappa central arm + Arb-stamped tails
.venv/bin/python scripts/assemble_erf_envelope.py

# 3. independent Arb cross-check of every committed eps
cargo run -p chelis-prove --features arb --bin certify_erf_envelope \
  -- validate crates/chelis-prove/data/erf_envelope.json

# 4. review and commit
git diff crates/chelis-prove/data/
```

## CI re-validation (the gate)

CI re-checks the committed artifacts without trusting them:

```sh
# re-run every committed Gappa proof + verify the bundle sha256
.venv/bin/python scripts/generate_erf_proof.py --check-only

# independently re-validate every committed eps against the Arb oracle
cargo run -p chelis-prove --features arb --bin certify_erf_envelope \
  -- validate crates/chelis-prove/data/erf_envelope.json
```

The Rust `arb` test lane additionally re-derives every box's `eps` from Arb
(`committed_envelope_eps_still_bounds_the_truth`), with a negative partner that
confirms a shrunk `eps` is caught, and locks the envelope to the proof bundle.
None of this touches the `default`/`smt` lanes, which neither link Arb nor
re-validate — they consume the committed data as-is.

## Environment

### Proof toolchain (Sollya + Gappa) — `scripts/setup_proof_toolchain.py`

Only three base libraries need root; everything else is built from source with
no sudo into a workspace-local `.local/` prefix.

| Need              | Fedora                | Debian/Ubuntu        |
|-------------------|-----------------------|----------------------|
| MPFI headers      | `mpfi-devel`          | `libmpfi-dev`        |
| libxml2 headers   | `libxml2-devel`       | `libxml2-dev`        |
| Gappa binary      | `gappa`               | `gappa`              |

```sh
# Fedora
sudo dnf install -y mpfi-devel libxml2-devel gappa
# Debian/Ubuntu
sudo apt-get install -y libmpfi-dev libxml2-dev gappa
```

`setup_proof_toolchain.py` then downloads (with pinned sha256) and builds
**fplll 5.4.5** and **Sollya 8.0** from source into `.local/`, linking the
system gmp/mpfr/mpfi/libxml2. Two portability fixes are applied automatically
because Sollya 8.0 predates the GCC 14/15 C23 default:

- `CFLAGS` carries `-std=gnu17` (under GCC 15's gnu23 default, Sollya's K&R
  empty-paren prototypes are hard `conflicting types` errors), and
- a one-line idempotent patch gives `miniyyparse` a prototype matching the
  bison-generated definition.

The build needs a C/C++ toolchain (`gcc`, `g++`, `make`, `m4`, `autoconf`,
`libtool`, `bison`, `flex`) and `gmp-devel`/`mpfr-devel`. Run sollya with
`LD_LIBRARY_PATH=.local/lib`.

Licensing (tracked, not a blocker): Sollya is CeCILL-C, fplll is LGPL-2.1+. Both
are offline build tooling that produce the committed proof bundle; neither links
into any shipped chelis binary.

### Arb certifier / cross-check — the `arb` cargo feature

`arb-sys` vendors and compiles FLINT/Arb (and `gmp-mpfr-sys` compiles
GMP/MPFR). It does **not** use a system FLINT.

| Tool       | Purpose                             | Install (Debian/Ubuntu) |
|------------|-------------------------------------|-------------------------|
| C compiler | compile vendored FLINT/Arb/GMP/MPFR | `apt install gcc`       |
| make, m4   | vendored build drivers              | `apt install make m4`   |

`.cargo/config.toml` bakes `CFLAGS=-fPIC` (Fedora PIE) and isolated
`GMP_MPFR_SYS_CACHE`/`FLINT_SYS_CACHE`/`ARB_SYS_CACHE` dirs so a clean
`--features arb` build is position-independent with no per-developer step. The
LGPL obligation attaches only to a binary built with the `arb` feature
(offline/CI tooling, never shipped); recorded and deferred per master WI-19.
