# Reef and packages

Reef manages Chelis project dependencies. The compiler-bundled `chelis-std`
runtime is available automatically; other libraries are declared in a
project's `reef.toml`.

## Start a project with a library

Create a project with `reef init`:

```sh
chelis reef init demo --module-prefix Demo --output demo
cd demo
```

Chelis libraries distributed through Reef are called shells; Nautilus
(numerical methods) is one. Install the release you need into your local Reef
registry. The release is public, so no token is needed:

```sh
chelis reef install --from-github Chelis-Lang/nautilus@v0.7.58
```

Then add it under `[dependencies]` in the project's `reef.toml`:

```toml
[dependencies]
nautilus = { version = "0.7.58" }
```

Each shell release pins an exact Chelis compiler version. The compiler pin in
your project must match the version required by the shell release; Reef
rejects a mismatch. For example, Nautilus 0.7.58 uses Chelis 0.22.1. Set the
generated `compiler` field in `reef.toml` to `"=0.22.1"`, then prepare and
build the project:

```sh
chelis reef setup
chelis reef build
```

`reef setup` installs the pinned compiler if it is missing. `reef build`
resolves the declared libraries from your local registry, records the selected
versions in `reef.lock`, and prints `Built demo 0.1.0`. You do not need to
check out the source repository of Nautilus to use it. Install every other
published library your project imports the same way, including the libraries
it depends on, and add it to the same `[dependencies]` table at a version
built for the project's compiler pin. If a declared
library is not in the local registry, `reef build` can discover its public
GitHub release after you set `GITHUB_TOKEN` or authenticate with
`gh auth login`.

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

## Fetch locked packages with another tool

A build system such as Nix can download the locked packages itself and give
them to Reef. `reef fetch-plan` prints the public download URL and the pinned
SHA-256 of each file that `reef.lock` locks. It reads only the lockfile and
makes no network request:

```sh
chelis reef fetch-plan --package-root .
```

For the hello-chelis 0.1.17 lockfile, the output starts like this:

```json
{
  "format_version": 1,
  "package": {
    "name": "hello-chelis",
    "version": "0.1.17"
  },
  "artifacts": [
    {
      "name": "chelis-clarabel",
      "version": "0.1.2",
      "kind": "archive",
      "file": "chelis-clarabel-0.1.2.tar.zst",
      "url": "https://github.com/chelis-lang/chelis-clarabel/releases/download/v0.1.2/chelis-clarabel-0.1.2.tar.zst",
      "sha256": "8291e653d051b2c53d8918419a10d38f072b2b7db5003d0b485eff2115e275e8",
      "remote_origin": "github://chelis-lang/chelis-clarabel@v0.1.2"
    },
```

Each package gives an `archive` and a `shell` element. A locked binary gives
one `binary` element with its `platform`. The bundled `chelis-std` and path
dependencies give no element. A lock entry without `remote_origin` makes the
command fail, because it has no download URL. The URLs serve public releases
only.

Download each `url`, check it against `sha256`, and put the file in one
directory under its `file` name. Then import the directory:

```sh
chelis reef install --from-lockfile --artifact-dir fetched
```

The import makes no network request. It compares each file with the
`reef.lock` hash before it writes to the registry. A missing file or a
different file fails its package, and the command then exits with an error.
Reef keeps the locked `remote_origin` of each package in its registry.

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

`chelis-std` comes from the compiler and is excluded from the bootstrap
input list. Dependency declarations follow the manifest schema: schema 1 uses
a bare exact version, while schemas 2 and 3 use requirements such as `=0.4.0`.
Each shell dependency must appear in the explicit bootstrap input list at a
version satisfying its requirement. A runtime requirement must be satisfied by
the compiler-bundled runtime.

To inspect or change a library's implementation, clone that library's source
repository. A source checkout is not needed to consume its Reef release.
