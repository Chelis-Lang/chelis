# EARS-to-Chelis Bridge

**Status:** v0.2 implementation spec.

**Owning package:** `c-earchin`.

`c-earchin` translates EARS requirement files into Chelis Deep witness
definitions plus a provenance sidecar. It follows the same external-notation
bridge pattern as Octant: source notation goes in, Deep plus spans comes out.

v0.2 targets Chelis Level 2 property discovery. It still does not add a new
Deep tag: each requirement emits an ordinary callable `bool` witness `def`.
The witness carries both stable `c_earchin_*` provenance metadata and canonical
Chelis property metadata so `chelis prove` can discover and run it.

## 1. Package Shape

`c-earchin` is a standalone adjacent shell repository:

```text
c-earchin/
├── Cargo.toml
├── reef.toml
├── src/
│   ├── lib.rs
│   ├── main.rs
│   ├── parser.rs
│   ├── classify.rs
│   ├── vocabulary_lookup.rs
│   ├── lower.rs
│   ├── spans.rs
│   ├── emit.rs
│   ├── vocabulary.ch
│   └── version.ch
├── references/
├── properties/
├── tests/
└── docs/
```

Rust source and Chelis source intentionally coexist under `src/`, matching the
Octant shell layout.

## 2. EARS File Format

EARS is a notation, so `c-earchin` pins a simple file format for v0.1.

- One requirement per block.
- A requirement starts at column 1 as `ID  requirement text`, with at least two
  spaces after `ID`.
- `ID` format: `[A-Za-z][A-Za-z0-9_-]*`.
- Indented following lines continue the current requirement.
- Blank lines separate blocks.
- Full-line `#` comments are allowed outside blocks.
- Continuation indentation is not semantic. The parser trims continuation
  indentation and joins wrapped lines with one ASCII space.

Example:

```text
PRC-001  The pricing engine shall produce non-negative call prices.
PRC-002  WHEN a tick arrives, the running mean shall update toward
         the tick value.
```

## 3. Generated Module And Witness Names

Default generated module:

```text
CEarchin.Generated.<FileStem>
```

`<FileStem>` is derived from the input filename:

- Strip only the final `.ears` extension.
- The remaining stem must match `[A-Za-z][A-Za-z0-9_-]*`.
- Hyphen/underscore-separated words convert to PascalCase.
- Dots, spaces, non-ASCII, and other punctuation are hard errors.

Examples:

- `pricing-rules.ears` -> `CEarchin.Generated.PricingRules`
- `pricing_v2.ears` -> `CEarchin.Generated.PricingV2`
- `pricing.v2.ears` -> error; rename to `pricing-v2.ears` or pass
  `--module`.

Witness name transform:

```text
transform(id) = "req_" + id.replace("-", "_")
```

No case folding and no other normalization occurs. `PRC-001` becomes
`req_PRC_001`; `prc-001` becomes `req_prc_001`. IDs that collide after this
transform, such as `PRC-001` and `PRC_001`, are hard errors and the diagnostic
must name both source locations.

## 4. Metadata Contract

All package-owned metadata keys use the `c_earchin_` prefix. The underscore is
the package-name shoreline for metadata keys: the package name is `c-earchin`,
but metadata keys cannot contain hyphens.

c-earchin-owned keys:

| Key | Value | Meaning |
|---|---|---|
| `c_earchin_role` | string enum | Closed role registry. v0.1 defines only `"property_witness"`. |
| `c_earchin_ears_id` | string | Source EARS requirement ID. |
| `c_earchin_ears_pattern` | string | `ubiquitous`, `event_driven`, `state_driven`, `unwanted`, `optional`, or `complex`. |
| `c_earchin_vocabulary_target` | string | Vocabulary predicate selected by lowering, or `"unresolved"` in non-strict miss mode. |
| `c_earchin_clause_spans` | string | Compact clause-span summary for humans; authoritative data lives in `.spans.json`. |

Future roles require this spec to change. Consumers must not infer an open-ended
role vocabulary from implementation code.

Canonical Chelis property keys emitted by v0.2 and later:

| Key | Value | Meaning |
|---|---|---|
| `chelis_role` | `"property"` | Marks the witness as a `chelis prove` property. |
| `property_source_kind` | `"bridge:c-earchin"` | Identifies the bridge producer. |
| `property_quantifiers` | `(params {})` | EARS v0.2 witnesses have no generated binders. |
| `property_preconditions` | `(tuple {})` | EARS v0.2 witnesses have no generated preconditions. |
| `property_source_id` | string | Source EARS requirement ID. |

The legacy `c_earchin_role: "property_witness"` key remains indefinitely for
bridge compatibility; Chelis accepts it as a discovery synonym.

## 5. EARS Patterns

v0.1 recognizes Mavin-style EARS patterns:

- Ubiquitous: `The [system] shall [response].`
- Event-driven: `WHEN [trigger], the [system] shall [response].`
- State-driven: `WHILE [state], the [system] shall [response].`
- Unwanted behavior: `IF [condition], THEN the [system] shall [response].`
- Optional: `WHERE [feature], the [system] shall [response].`
- Complex: combinations of the above clauses before `the [system] shall`.

Out-of-scope forms must fail with source locations rather than silently
lowering to empty output.

## 6. Compatibility Property Shape

Each requirement lowers to one callable witness definition returning `bool`.
The body is not a placeholder `true` when vocabulary lookup succeeds; it calls
the selected `CEarchin.Vocabulary` predicate with v0.1 placeholder arguments.
This makes the witness callable today while leaving full property harnessing to
`chelis prove`.

When vocabulary lookup misses:

- `--strict` fails.
- Non-strict mode emits a callable witness returning `true` with
  `c_earchin_vocabulary_target: "unresolved"`.

## 7. Vocabulary Surface

`src/vocabulary.ch` owns `CEarchin.Vocabulary`. Scaffold commits must create the
complete v0.1 predicate signature set. Later implementation commits flesh out
bodies without changing signatures.

v0.1 predicates:

- `is_non_negative(x: f32) -> bool`
- `is_positive(x: f32) -> bool`
- `within_tolerance(x: f32, y: f32, tol: f32) -> bool`
- `is_monotone_increasing(x: f32, x_prime: f32, fx: f32, fx_prime: f32) -> bool`
- `is_bounded_above(x: f32, bound: f32) -> bool`
- `is_bounded_below(x: f32, bound: f32) -> bool`
- `is_equal(x: f32, y: f32) -> bool`
- `converges_to(x: f32, target: f32, tol: f32) -> bool`
- `is_error_rejection() -> bool`

Tests must fail if an emitted vocabulary reference has no corresponding
definition.

## 8. CLI

```text
c-earchin translate INPUT [--output OUTPUT] [--spans SPANS]
                         [--inline] [--no-spans] [--strict]
                         [--module MODULE]
c-earchin check INPUT
c-earchin explain SPANS_JSON --target <deep-node-id>
c-earchin version
```

`explain` v0.1 reads the `.spans.json` manifest directly and linearly scans
`spans[]`. Indexed spans are future work.

## 9. Acceptance Oracle

The v0.1 oracle is:

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
chelis validate --deep <every generated .dp golden>
chelis prove references/vocabulary_miss.dp --spans references/vocabulary_miss.spans.json --json
```

A fresh-context red team is required after spec lock, after translator core,
and before release.
