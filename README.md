# Chelis

Chelis is a programming language for tensor computation, designed to be written by both humans and AI agents. It compiles through a minimal set of ~12 primitive operations (the RISC DAG) to efficient C code, with automatic differentiation as a first-class language feature.

**Status:** Phase 0f (C backend codegen) in progress. Phases 0a-0e complete. All core spec documents written and reviewed.

## Build

```
cargo build --workspace
cargo test --workspace
```

## Project Structure

```
crates/
  chelis-deep/       Deep (s-expression) parser
  chelis-surf/       Surface syntax parser + desugaring
  chelis-types/      Type checker (HM inference, named dims, precision)
  chelis-ir/         RISC DAG intermediate representation
  chelis-backend-c/  C code generation backend
  chelis-cli/        CLI binary
spec/                Language specification
examples/            Example Chelis programs (coming soon)
```

## Key Ideas

- **Dual syntax:** Human-friendly surface syntax (Surf, `.ch`) desugars to canonical s-expressions (Deep, `.dp`)
- **Named tensor dimensions:** `tensor[batch, hidden, f32]` — dimensions are types, not positions
- **No implicit anything:** No silent type promotion, no implicit broadcasting, no hidden allocations
- **Compiler as training signal:** Type errors produce fitness scores (0.0-1.0) and repair suggestions, not just error messages
- **~12 RISC primitives:** Every operation — matmul, softmax, conv2d — decomposes to a small set of primitive ops
- **`grad` is a language feature:** Reverse-mode AD as a DAG-to-DAG rewrite, not a library

## Documentation

- [Architecture Guide](ARCHITECTURE.md) — How the compiler works
- [Specification](spec/) — Language spec (start with [00-context.md](spec/00-context.md))

## License

MIT
