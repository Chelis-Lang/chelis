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
{"kind":"DimensionMismatch","message":"tensor rank mismatch: 1 dims vs 2 dims","severity":0.8,"span":{"span":"point","offset":94},"span_id":"surf:94..103"}
```

- **Checked before it runs.** The compiler checks shapes, precision, effects, and
  ownership. Dimensions match by name, `f32` and `f64` do not mix without a `cast`,
  and randomness and I/O appear in a function's type.
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

## Install a release toolchain

Chelis releases include `chelisup`, which installs toolchains and selects the
version used by each project. `chelisup` downloads release assets through the
authenticated GitHub REST API, so it needs a GitHub token even though the
releases are public; an authenticated [GitHub CLI](https://cli.github.com)
provides one.

```sh
gh auth login
gh release download --repo Chelis-Lang/chelis --pattern chelisup.sh --output - | sh
# If the bootstrap says ~/.chelis/bin is not on PATH:
export PATH="$HOME/.chelis/bin:$PATH"
release_tag="$(gh release view --repo Chelis-Lang/chelis --json tagName --jq .tagName)"
chelisup install "${release_tag#v}"
chelis --version
```

The bootstrap script installs `chelisup`; the next command installs the
compiler. The release tag has a leading `v`, while `chelisup install` takes
the bare `X.Y.Z` version. For a project with a `reef.toml`, install the
version named by its `compiler` pin before running `chelis reef setup`, then
run `chelis reef build`. The [install guide](docs/book/src/install.md)
explains the project workflow.

## Start here

- [Chelis Guide](docs/book/src/README.md): first program, CLI, properties, and Reef.
- [Examples](examples/): executable Chelis programs.
- [Language spec](spec/00-context.md) and the
  [canonical project reference](spec/design/chelis_canonical_reference.md).
- [Architecture](ARCHITECTURE.md): compiler, evaluator, and code generation.
- [Contributor setup](docs/contributor_setup.md), the
  [contribution guide](CONTRIBUTING.md), and the [agent contract](AGENTS.md).

## License

MIT
