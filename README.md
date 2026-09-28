# Chelis

Chelis is a functional language for AI research. It is designed for programs
such as models, training loops, and learned functions, with a coding agent as
the primary author and a human as supervisor. Surf (`.ch`) is the readable
syntax; Deep (`.dp`) is the canonical syntax used by the compiler and agents.

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

The release tag has a leading `v`; `chelisup install` takes the bare `X.Y.Z`
version. In a project with a `reef.toml`, run `chelis reef setup` to install
its pinned toolchain and dependencies. The [install guide](docs/book/src/install.md)
explains version selection and setup.

## Start here

- [User book](docs/book/src/README.md): first program, CLI, Reef, and backends.
- [Examples](examples/): executable Chelis programs.
- [Architecture](ARCHITECTURE.md): compiler, evaluator, and code generation.
- [Contributor setup](docs/contributor_setup.md) and [contribution guide](CONTRIBUTING.md):
  build from source and prepare a change.
- [Agent contract](AGENTS.md): repository rules for coding agents.
- [Canonical project reference](spec/design/chelis_canonical_reference.md) and
  [numbered language specs](spec/00-context.md): project intent and language
  contracts.

The repository also contains `chelis-std`, the runtime bundled with the
compiler, and Reef, the package system for downstream shells.

## License

MIT
