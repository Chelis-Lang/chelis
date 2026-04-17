# Reef and Packages

Reef is Chelis' package and shell distribution layer.

## What Lives in Reef

- `chelis-std` is the standard library shell
- downstream shells such as Nautilus are packaged separately
- applications pin an exact compiler version in `reef.toml`

## Package Manifest

```toml
[package]
name = "demo"
version = "0.1.0"
compiler = "=0.1.7"
module_prefix = "Demo"
```

## Import Syntax

```chelis-surf-fragment
import Std.Nn.Linear(..)
```

## Typical Workflow

```sh
chelis reef init demo --module-prefix Demo --output demo
chelis reef build demo
```

For the current release contract and downstream pinning policy, use
`spec/design/phase3j_pre_release.md`.
