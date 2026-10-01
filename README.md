# Chelis

Chelis is a functional language for AI research. It is designed for programs
such as models, training loops, and learned functions. Surf (`.ch`) is the
source syntax; Deep (`.dp`) is the compiler's canonical representation.

<p align="center">
  <img src="assets/mascot/chev.svg" alt="Chev Chelis, the project mascot — a turtle on a mountain bike climbing a hill" width="320"/>
</p>

## Install a release toolchain

Chelis releases include `chelisup`, which installs toolchains and selects the
version used by each project. Access to the private repository and an
authenticated [GitHub CLI](https://cli.github.com) are currently required.

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

- [Chelis Guide](docs/book/src/README.md): first program, language reference,
  CLI, packages, and backends.
- [Examples](examples/): executable Chelis programs.
- [Architecture](ARCHITECTURE.md): compiler, evaluator, and code generation.
- [Contributor setup](docs/contributor_setup.md) and [contribution guide](CONTRIBUTING.md):
  build from source and prepare a change.
- [Language specifications](spec/00-context.md): the detailed contracts for
  syntax, types, operations, and targets.

The toolchain bundles the `chelis-std` library and the native runtime used by
generated code. Reef manages project dependencies.

## License

MIT
