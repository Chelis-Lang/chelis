# Chelis for VS Code

The extension provides syntax highlighting for Surf (`.ch`) and Deep (`.dp`)
files and connects the editor to the Tide language server.

Surf files receive live diagnostics, completion, hover information, and
definition lookup. Deep files receive syntax highlighting and diagnostics. The
status bar shows the Surf document's fitness score, and **Chelis: Show Deep**
opens its read-only canonical Deep form.

## Run from a Chelis checkout

Open the checkout in VS Code's Extension Development Host:

```sh
code --extensionDevelopmentPath="$PWD/editors/vscode" "$PWD"
```

Open a `.ch` or `.dp` file in the new window. The extension starts
`chelis tide lsp` over standard input and output. In a Chelis checkout it uses
`target/debug/chelis` when that binary exists, or runs the server through Cargo.
In another workspace it starts `chelis tide lsp` from `PATH`.

The server and extension logs appear in **View → Output**, under **Chelis**.
