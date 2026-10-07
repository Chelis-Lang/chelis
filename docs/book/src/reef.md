# Reef and packages

Reef manages Chelis project dependencies. The compiler-bundled `chelis-std`
runtime is available automatically; other libraries are declared in a
project's `reef.toml`.

## Start a project with a library

Create a project with `reef init`, then add the library you need under
`[dependencies]` in its `reef.toml`:

```sh
chelis reef init demo --module-prefix Demo --output demo
cd demo
```

```toml
[dependencies]
nautilus = { version = "0.7.50" }
```

Shell releases pin an exact Chelis compiler version. The compiler pin in your
project must match the version required by the shell release; Reef rejects a
mismatch. For example, Nautilus 0.7.50 uses Chelis 0.19.1. Set the generated
`compiler` field in `reef.toml` to `"=0.19.1"`, then prepare and build the
project:

```sh
chelis reef setup
chelis reef build
```

Reef installs the pinned compiler, downloads the declared library from its
release, and resolves its dependencies. It records the selected versions in
`reef.lock`. You do not need to check out the source repository of Nautilus or
its dependencies to use them. Add other published libraries your project
imports to the same `[dependencies]` table, using versions compatible with the
project's compiler pin.

## Set up an existing project

For a project that already has a `reef.toml` and `reef.lock`, provision its
pinned compiler and locked packages, then build:

```sh
chelis reef setup
chelis reef build
```

`reef setup` uses the project's lockfile to retrieve its dependencies. This is
the usual way to prepare an application such as
[hello-chelis](https://github.com/Chelis-Lang/hello-chelis) on another machine.
Clone that application to access its example programs; Reef downloads the
libraries it depends on separately.

## Create a package

To start a new library or application, use `reef init`:

```sh
chelis reef init demo --module-prefix Demo --output demo
cd demo
chelis reef build
```

The command creates a `reef.toml` manifest and a starter module under `src/`.
The module name follows the manifest's `module_prefix` and its path. For
example, `src/nn/linear.ch` in this project uses a module name under
`Demo.Nn.Linear`.

## About bulk shell installation

`chelis reef install --bootstrap` downloads a predefined set of commonly used
shells. It is not a complete list of every published library. For application
development, declare the libraries your program uses in `reef.toml`; Reef then
fetches that dependency set and its transitive dependencies, pinned by
`reef.lock`.

To inspect or change a library's implementation, clone that library's source
repository. A source checkout is not needed to consume its Reef release.
