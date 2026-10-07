# Chelis

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/brand/chelis-banner-dark.svg">
    <img src="docs/assets/brand/chelis-banner-light.svg" alt="Line drawing of a green sea turtle swimming over kelp" width="100%">
  </picture>
</p>

Chelis is a numerical computing language for code that agents write and people
supervise. Tensors carry named dimensions and precision in their type, and a proof
stack checks the properties you state.

```chelis
def portfolio_return(w: tensor[3, f64], r: tensor[3, 1, f64]) -> tensor[f64] = {
  weighted = mul(w, r)
  weighted |> sum(0)
}
```

In numpy, returns shaped 3 by 1 stretch against three weights into a 3 by 3 grid and
the sum comes back as a plausible wrong number. `chelis check` rejects the same
multiply before anything runs. The error from its JSON report:

```json
{"kind":"DimensionMismatch","message":"`mul` argument 2: expected rank-1 tensor, got rank-2 tensor","severity":0.8,"expected":"rank-1 tensor","got":"rank-2 tensor","span":{"span":"point","offset":94},"span_id":"surf:94..103"}
```

- **Checked before it runs.** The compiler checks shapes, precision, effects, and
  ownership. Dimensions match by name, `f32` and `f64` do not mix without a `cast`,
  I/O appears in a function's effects, and random operations take explicit keys.
- **An agent in the loop.** `chelis check` answers in JSON with the error kind and
  source span, and the same input always gets the same answer. `chelis tide mcp`
  gives an agent check, eval, prove, and structural edits as MCP tools.
- **Properties you can review.** `chelis prove` checks `@property` declarations with
  type checking, an SMT solver, or seeded sampling, and each result names the
  method behind it. A second checker written in Chelis cross-checks the compiler,
  and a core calculus of Chelis is mechanized in Lean 4.
- **General-purpose numerics.** Surf (`.ch`) is the readable syntax; Deep (`.dp`) is
  the canonical form the compiler and agents use. Programs build to C. Shells,
  installed with Reef, cover numerical methods (Nautilus), dataframes (Coral), and
  quantitative finance (Shoals).

## Install

Chelis is distributed as prebuilt binaries for Linux x86_64 and macOS arm64,
published on the [releases](https://github.com/Chelis-Lang/chelis/releases)
page. The [install guide](https://chelis.ch/docs/chelis/install/) has the
steps and the native C compiler `chelis build` needs on each platform.

## Documentation

The [Chelis guide](https://chelis.ch/docs/chelis/) covers the first program,
the CLI, types, effects, properties, and Reef packages. The same guide is the
mdBook in [`docs/book`](docs/book/), rendered from chelis.ch; build it locally
with `mdbook build docs/book`. The [examples](examples/) directory holds executable
Chelis programs.

## License

MIT
