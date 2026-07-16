# Clippy Pedantic Cleanup Inventory

Status: baseline inventory captured 2026-07-16.

## Scope and method

The workspace currently has no `[workspace.lints.clippy]` configuration, and the
canonical gate runs Clippy with `-D warnings` but does not enable the
allow-by-default `clippy::pedantic` group. A source scan found 80 explicit
`allow(clippy::...)` entries, but none suppress a member of the pedantic group.
The debt is therefore a workspace-level enforcement gap rather than a set of
local pedantic exemptions.

The baseline used stable Cargo 1.97.0 and Clippy 0.1.97 and was collected with:

```text
CARGO_TARGET_DIR=target/agents/inventory \
  cargo clippy --workspace --all-targets --message-format=json -- \
  --force-warn clippy::pedantic
```

`--force-warn` ensures that future local attributes cannot hide findings from
this inventory. The JSON diagnostics were deduplicated by lint, message, and
source span after resolving standard-library macro spans back to their workspace
call sites. The command covers all workspace targets with default features; it
does not cover every optional feature combination. Findings in `src/` include
inline test modules.

## Baseline

- Raw diagnostics, including all-target duplicate replays: **9,100**
- Unique source findings: **5,766**
- Duplicate replays removed: **3,334**
- Distinct pedantic lints: **67**
- Crates or trees affected: **28**
- Findings with a machine-applicable suggestion: **3,276**
- Findings requiring manual review or lacking a machine-applicable suggestion:
  **2,490**

### By source surface

| Count | Surface |
|---:|---|
| 3,894 | `src/`, including inline tests |
| 1,867 | Explicit test paths |
| 3 | Build scripts |
| 2 | Other |

### By lint

| Count | Lint |
|---:|---|
| 922 | `clippy::needless_raw_string_hashes` |
| 686 | `clippy::doc_markdown` |
| 528 | `clippy::must_use_candidate` |
| 327 | `clippy::cast_possible_truncation` |
| 277 | `clippy::map_unwrap_or` |
| 227 | `clippy::missing_errors_doc` |
| 216 | `clippy::cast_precision_loss` |
| 215 | `clippy::cast_sign_loss` |
| 188 | `clippy::uninlined_format_args` |
| 186 | `clippy::float_cmp` |
| 167 | `clippy::match_same_arms` |
| 163 | `clippy::too_many_lines` |
| 142 | `clippy::cast_lossless` |
| 142 | `clippy::unreadable_literal` |
| 133 | `clippy::redundant_closure_for_method_calls` |
| 98 | `clippy::items_after_statements` |
| 96 | `clippy::manual_let_else` |
| 89 | `clippy::cast_possible_wrap` |
| 76 | `clippy::needless_pass_by_value` |
| 67 | `clippy::unnecessary_literal_bound` |
| 62 | `clippy::similar_names` |
| 57 | `clippy::missing_panics_doc` |
| 45 | `clippy::format_push_string` |
| 44 | `clippy::single_match_else` |
| 42 | `clippy::unnested_or_patterns` |
| 41 | `clippy::cast_ptr_alignment` |
| 37 | `clippy::semicolon_if_nothing_returned` |
| 37 | `clippy::many_single_char_names` |
| 33 | `clippy::manual_assert` |
| 32 | `clippy::if_not_else` |
| 32 | `clippy::ptr_cast_constness` |
| 28 | `clippy::case_sensitive_file_extension_comparisons` |
| 27 | `clippy::used_underscore_binding` |
| 27 | `clippy::ignore_without_reason` |
| 26 | `clippy::match_wildcard_for_single_variants` |
| 25 | `clippy::ptr_as_ptr` |
| 24 | `clippy::unused_self` |
| 23 | `clippy::assigning_clones` |
| 21 | `clippy::wildcard_imports` |
| 17 | `clippy::implicit_hasher` |
| 15 | `clippy::default_trait_access` |
| 14 | `clippy::return_self_not_must_use` |
| 13 | `clippy::single_char_pattern` |
| 10 | `clippy::implicit_clone` |
| 8 | `clippy::unnecessary_wraps` |
| 8 | `clippy::bool_to_int_with_if` |
| 7 | `clippy::duration_suboptimal_units` |
| 7 | `clippy::unnecessary_debug_formatting` |
| 7 | `clippy::unnecessary_trailing_comma` |
| 7 | `clippy::explicit_iter_loop` |
| 6 | `clippy::trivially_copy_pass_by_ref` |
| 5 | `clippy::stable_sort_primitive` |
| 5 | `clippy::elidable_lifetime_names` |
| 4 | `clippy::no_effect_underscore_binding` |
| 4 | `clippy::ref_as_ptr` |
| 4 | `clippy::borrow_as_ptr` |
| 3 | `clippy::struct_field_names` |
| 3 | `clippy::ref_option` |
| 2 | `clippy::ignored_unit_patterns` |
| 2 | `clippy::needless_continue` |
| 1 | `clippy::fn_params_excessive_bools` |
| 1 | `clippy::enum_glob_use` |
| 1 | `clippy::collapsible_else_if` |
| 1 | `clippy::manual_is_variant_and` |
| 1 | `clippy::manual_string_new` |
| 1 | `clippy::self_only_used_in_recursion` |
| 1 | `clippy::option_option` |

### By crate or tree

| Count | Crate or tree | Top lints |
|---:|---|---|
| 960 | `chelis-ir` | `doc_markdown` (199), `needless_raw_string_hashes` (165), `map_unwrap_or` (81), `must_use_candidate` (69), `match_same_arms` (61) |
| 859 | `chelis-types` | `needless_raw_string_hashes` (339), `must_use_candidate` (73), `doc_markdown` (69), `map_unwrap_or` (40), `manual_let_else` (35) |
| 614 | `chelis-cli` | `doc_markdown` (149), `needless_raw_string_hashes` (145), `map_unwrap_or` (54), `redundant_closure_for_method_calls` (44), `cast_precision_loss` (19) |
| 607 | `chelis-compiler-api` | `needless_raw_string_hashes` (133), `cast_possible_truncation` (83), `missing_errors_doc` (44), `cast_sign_loss` (37), `cast_lossless` (34) |
| 501 | `chelis-prove` | `must_use_candidate` (124), `unreadable_literal` (53), `doc_markdown` (52), `cast_precision_loss` (39), `missing_errors_doc` (28) |
| 403 | `chelis-runtime` | `cast_sign_loss` (103), `cast_possible_truncation` (80), `cast_ptr_alignment` (41), `cast_possible_wrap` (39), `ptr_cast_constness` (32) |
| 318 | `chelis-e2e` | `float_cmp` (103), `cast_precision_loss` (77), `needless_raw_string_hashes` (37), `cast_lossless` (12), `cast_possible_truncation` (11) |
| 295 | `chelis-backend-c` | `doc_markdown` (74), `cast_possible_truncation` (29), `cast_lossless` (24), `must_use_candidate` (17), `match_same_arms` (16) |
| 230 | `chelis-backend-hip` | `doc_markdown` (61), `must_use_candidate` (56), `semicolon_if_nothing_returned` (22), `cast_possible_truncation` (17), `match_same_arms` (16) |
| 160 | `chelis-reef` | `missing_errors_doc` (43), `uninlined_format_args` (38), `redundant_closure_for_method_calls` (17), `doc_markdown` (13), `too_many_lines` (12) |
| 148 | `chelis-deep` | `uninlined_format_args` (63), `must_use_candidate` (23), `missing_errors_doc` (19), `needless_raw_string_hashes` (9), `manual_let_else` (7) |
| 136 | `chelis-lint` | `unnecessary_literal_bound` (60), `needless_raw_string_hashes` (27), `doc_markdown` (15), `must_use_candidate` (10), `redundant_closure_for_method_calls` (6) |
| 125 | `chelis-backend-metal` | `must_use_candidate` (35), `ignore_without_reason` (19), `doc_markdown` (12), `missing_panics_doc` (7), `cast_possible_truncation` (7) |
| 88 | `chelis-surf` | `map_unwrap_or` (14), `must_use_candidate` (13), `too_many_lines` (10), `match_same_arms` (10), `doc_markdown` (5) |
| 51 | `chelis-conformance` | `must_use_candidate` (29), `redundant_closure_for_method_calls` (5), `missing_errors_doc` (5), `map_unwrap_or` (4), `manual_assert` (2) |
| 51 | `chelis-effects` | `needless_raw_string_hashes` (32), `doc_markdown` (6), `missing_errors_doc` (3), `redundant_closure_for_method_calls` (3), `uninlined_format_args` (2) |
| 51 | `chelis-tide` | `needless_raw_string_hashes` (15), `items_after_statements` (12), `needless_pass_by_value` (6), `missing_errors_doc` (4), `doc_markdown` (3) |
| 50 | `chelis-python` | `cast_possible_wrap` (12), `cast_possible_truncation` (10), `cast_sign_loss` (8), `case_sensitive_file_extension_comparisons` (3), `redundant_closure_for_method_calls` (3) |
| 33 | `chelisup` | `must_use_candidate` (15), `missing_errors_doc` (5), `single_match_else` (3), `redundant_closure_for_method_calls` (3), `manual_let_else` (2) |
| 27 | `chelis-lsp` | `must_use_candidate` (9), `map_unwrap_or` (5), `missing_errors_doc` (3), `too_many_lines` (3), `needless_pass_by_value` (2) |
| 17 | `chelis-cove` | `match_same_arms` (9), `map_unwrap_or` (4), `cast_possible_truncation` (2), `missing_errors_doc` (1), `needless_pass_by_value` (1) |
| 14 | `chelis-macros` | `needless_raw_string_hashes` (6), `must_use_candidate` (3), `redundant_closure_for_method_calls` (2), `missing_errors_doc` (1), `too_many_lines` (1) |
| 11 | `chelis-pred` | `must_use_candidate` (4), `match_same_arms` (2), `map_unwrap_or` (2), `single_match_else` (1), `missing_errors_doc` (1) |
| 5 | `chelis-std-bundle` | `must_use_candidate` (2), `manual_assert` (1), `missing_errors_doc` (1), `redundant_closure_for_method_calls` (1) |
| 4 | `chelis-shell` | `missing_errors_doc` (4) |
| 4 | `chelis-validate` | `missing_errors_doc` (3), `needless_pass_by_value` (1) |
| 2 | `chelis-version` | `must_use_candidate` (2) |
| 2 | `tree-sitter-chelis` | `must_use_candidate` (2) |

## Cleanup order

1. Add the gate-command expectation before changing the gate implementation, per
   the repository's spec-first rule.
2. Apply and review low-risk, machine-applicable rewrites one lint at a time,
   starting with `needless_raw_string_hashes`, `uninlined_format_args`, and
   `redundant_closure_for_method_calls`.
3. Handle documentation and API-contract lints separately. In particular,
   `must_use_candidate`, `missing_errors_doc`, and `missing_panics_doc` require
   public-surface judgment rather than blind autofix.
4. Handle numeric, pointer, and FFI cast lints with targeted positive and
   negative tests for range, sign, precision, and alignment assumptions.
5. Review structural lints such as `too_many_lines`, `similar_names`, and
   `many_single_char_names` without hiding them behind broad workspace allows.
6. Enable pedantic denial in the canonical gate only after the inventory reaches
   zero, then keep any necessary local exemption narrow and reasoned.

## Completion oracle

The authoritative completion oracle is:

```text
python3 scripts/gate.py --local
```

Completion additionally requires that the gate's mechanically tested Clippy
command enables `clippy::pedantic`; otherwise a green result does not prove this
cleanup.
