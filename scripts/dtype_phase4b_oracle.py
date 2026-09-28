#!/usr/bin/env python3
"""Authoritative chelis#729 Phase 4B semantic/schema freeze oracle.

Acceptance is exit 0 with the final line ``DTYPE PHASE 4B ORACLE: PASS``.

Usage:

    .venv/bin/python scripts/dtype_phase4b_oracle.py

Required atom identities and unambiguous contract-region boundaries are reviewed
literal declarations. Required semantic clauses, schema checks and fresh
source/compiler agreement remain independent guards. Text changes are named
by the committed report and require exact file/atom/region acknowledgements in
PR mode. No atom or region text digest needs recomputing. Acknowledgement is a
review obligation, not permission to omit a required clause or identity.

Acknowledgement gate
--------------------

For every path in ``CONTRACT_FILES`` the oracle compares the working-tree bytes
against the bytes at the merge base with the base branch. Any contract file
whose content differs must be acknowledged by name. The acknowledgement is a
line in the pull request body::

    Frozen-contract-change: spec/04-type-system.md

one line per changed file. The committed change report additionally requires
one line for each changed atom and region::

    Frozen-contract-change: atom:05-OP-33
    Frozen-contract-change: region:"agent numeric-surface discipline"

Region labels are JSON strings; atoms use their exact unbracketed identity.
File lines remain required even when the same edit changes an atom or region,
or removes its file from the contract inventory.
Registry-only edits require each owning atom, and removed identities or changed
protection/region declarations remain visible through both inventories. A file
line never substitutes for an identity line. Missing, duplicate, stale, unknown
or malformed addresses fail the enforcing mode. JSON escape spellings decode
to one region identity, so alternate spellings cannot evade duplicate checks.

The grammar is case-sensitive: no leading whitespace, exactly one space after
the colon, and no trailing content. File addresses are repo-relative POSIX
paths without glob metacharacters or `.`/`..` segments. The file leg compares
working-tree bytes; the identity leg compares committed snapshots. PR execution
validates the synthetic merge against `--pr-head` and uses its first parent,
the same comparison that produced the report artifact.

Lines inside fenced code blocks are ignored so a body can quote the grammar.
That exemption is pragmatic, not a CommonMark implementation: it tracks the
opening run's character and length, and it can still disagree with GitHub's
renderer in both directions (an HTML comment hides a line from a reader but not
from this parser; an indented fence hides it from this parser but not always
from a reader). The disagreement costs reviewer visibility, never soundness: no
shape of it admits an unacknowledged change, because a line the parser does not
read is a file that goes unacknowledged and fails.

``--require-acknowledgement`` is this oracle's local enforcing mode. The
dedicated ``PR Contract Acknowledgements`` check applies the same grammar and
change report through ``phase4b_change_report.py``; the Docs job runs this full
oracle independently. Without the flag the oracle reports the changed contract
files and the exact lines the body must carry, then exits 0, so a local run is a
checklist rather than a gate. An unresolvable merge base fails the enforcing
mode loudly; it can never be read as "nothing changed".

The acknowledgement gate first replaced whole-file SHA-256 digests. That table made two pull
requests that edited *different* contract files conflict on adjacent lines of
one Python dict, and two that edited the *same* file conflict on one line whose
correct post-rebase value is the digest of the merged text, so the conflict was
unresolvable by picking a side. The acknowledgement lives in the pull request
body, which no other pull request shares. Deliberateness is preserved: naming a
frozen contract file in the body is the same review-visible act that moving a
digest was, and it still owes the owning spec/design update, every consuming
contract, and an adversarial mutation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import subprocess
import sys


from phase4b_contract_text import (
    ATOM_START, OracleError, frozen_region, strict_atom_block,
)
from phase4b_change_report import (
    acknowledgement_line, changed_contracts, identity_acknowledgement_violations,
    resolve_comparison,
)
from phase4b_acknowledgements import (
    ACKNOWLEDGEMENT_KEY,
    acknowledgement_identity,
    parse_acknowledgements,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PASS_LINE = "DTYPE PHASE 4B ORACLE: PASS"
CONTRACT_FILES = (
    "AGENTS.md",
    "spec/02-surf-syntax.md",
    "spec/03-deep-syntax.md",
    "spec/04-type-system.md",
    "spec/05-risc-primitives.md",
    "spec/06-transformations.md",
    "spec/10-serialization.md",
    "spec/11-ffi.md",
    "spec/design/capability_table.md",
    "spec/design/compiled_value_ownership.md",
    "spec/design/dtype_semantics.md",
    "spec/design/implicit_linearity.md",
    "spec/design/loud_unsupported.md",
    "spec/design/spec_provenance.md",
    "spec/design/remediation_roadmap.md",
    "spec/design/runtime_extents.md",
    "spec/design/runtime_representation.md",
    "docs/CHELIS_SURFACE.md",
    "docs/investigations/remediation_status_2026_08_04.md",
    "spec/registry/builtin_semantic_identities.md",
    "spec/registry/c_scalar_carrier.md",
    "spec/registry/c_container_boundary.md",
    "spec/registry/c_tensor_runtime.md",
    "spec/registry/c_heap_lifetime.md",
    "spec/registry/stdlib_adt_identities.md",
    "spec/registry/stdlib_numeric_manifest.md",
    "spec/registry/python_tensor_metadata.md",
)
OP_ATOM = re.compile(r"^> \*\*\[05-OP-(\d+)\]\*\*", re.MULTILINE)
EXPECTED_PHASE4B_OP_HEADINGS = {
    1: "`round_to(x, places) -> r`",
    2: "Ingestion preserves",
    3: "`io/json::json_int`",
    4: "`JsonFloat(value)`",
    5: "`io/json::to_json`",
    6: "`cast_trunc(source, target)`",
    7: "The runtime extent read",
    8: "`uniform_like(k, template, low, high) -> result`",
    9: "`pad_sequences(sequences: List[List[T]], pad: T) ->",
    10: "`pad_sequences_to(sequences: List[List[T]], width: i64,",
    11: "`mean(x, axes...) -> result`",
    12: "`max_reduce(x, axes...) -> result`",
    13: "`min_reduce(x, axes...) -> result`",
    14: "`prod_reduce(x, axes...) -> result`",
    15: "`argmax_reduce(x, axis) -> result`",
    16: "`argmin_reduce(x, axis) -> result`",
    17: "`wrap_add(left, right) -> result`",
    18: "`wrap_sub(left, right) -> result`",
    19: "`wrap_mul(left, right) -> result`",
    20: "`is_nan(x) -> result`",
    21: "`is_finite(x) -> result`",
    22: "`is_infinite(x) -> result`",
    23: "`cast_saturate(source, target) -> result`",
    24: "`cast_wrap(source, target) -> result`",
    25: "`to_string(value) -> result`",
    26: "`and(left, right) -> result`",
    27: "`or(left, right) -> result`",
    28: "`not(value) -> result`",
    29: "`count(x, axes...) -> result`",
    30: "`sum(x, axes..., accumulator = default(p)) -> result`",
    31: "`scalar_carrier(value) -> result`",
    32: "`shape_index(container, parameters...) -> result`",
    33: "`runtime_tensor(value, parameters...) -> result`",
    34: "`numeric_adt(fields...) -> value`",
    35: "`stdlib_numeric_def(arguments...) -> result`",
    36: "`comparison(left, right) -> result`",
    37: "`dropout(k, input, rate) -> result`",
    38: "`host_numeric_builtin(arguments...) -> result`",
    39: "`window_reduction(arguments...) -> result`",
    40: "`max_elem(left, right) -> result` and",
    41: "`sub(left, right) -> result`",
    42: "`stop_gradient(value) -> result`",
    43: "`relu(x) -> result`",
    44: "`heap_lifetime(handle, parameters...) -> result`",
    45: "`python_tensor_shape(tensor) -> extents`",
}

# These are independent, executable copies of the exact normative manifests.
# The full-file and atom digests below make additive prose tamper-evident, while
# these rows make a missing, renamed, duplicated, or retyped callable explain
# itself as a manifest failure rather than only as an opaque hash mismatch.
EXPECTED_OP_MANIFESTS = {
    "05-OP-45": (
        "| full tensor shape | `chelis_python::NativeTensor::shape(self: &Self) -> Vec<i64>` |",
    ),
    "05-OP-31": tuple(
        """\
| dtype storage size | `int64_t chelis_dtype_size(chelis_dtype dtype)` |
| scalar validation/construction | `chelis_scalar chelis_scalar_from_bits(chelis_dtype dtype, uint64_t bits)` |
| value boxing | `chelis_value chelis_value_box_scalar(chelis_scalar value)` |
| value extraction | `chelis_scalar chelis_value_unbox_scalar(chelis_value value)` |
| rank-zero tensor construction | `chelis_tensor *chelis_scalar_tensor(chelis_scalar value)` |
| rank-zero tensor extraction | `chelis_scalar chelis_tensor_to_scalar(const chelis_tensor *tensor)` |
| tensor fill | `void chelis_fill_scalar(chelis_tensor_write *guard, chelis_scalar value)` |
| scalar rendering | `chelis_string chelis_string_from_scalar(chelis_scalar value)` |
| scalar parsing | `chelis_option *chelis_parse_scalar(chelis_string text, chelis_dtype dtype)` |
| exact dictionary scalar lookup | `chelis_option *chelis_dict_get_scalar(const chelis_dict *dict, chelis_value key, chelis_dtype dtype)` |""".splitlines()
    ),
    "05-OP-32": tuple(
        """\
| length-aware string construction | `chelis_string chelis_string_from_utf8(const uint8_t *value, int64_t len)` |
| character code | `int64_t chelis_char_code(chelis_string value)` |
| character from code | `chelis_string chelis_char_from_code(int64_t value)` |
| string print | `void chelis_print_string(chelis_string value)` |
| string length | `int64_t chelis_string_len(chelis_string value)` |
| string slice | `chelis_string chelis_string_slice(chelis_string value, int64_t start, int64_t len)` |
| list length | `int64_t chelis_list_len(const chelis_list *list)` |
| list construction | `chelis_list *chelis_list_from_values(const chelis_value *items, int64_t len)` |
| list index | `chelis_value chelis_list_index(const chelis_list *list, int64_t index)` |
| list take | `chelis_list *chelis_list_take(const chelis_list *list, int64_t count)` |
| list drop | `chelis_list *chelis_list_drop(const chelis_list *list, int64_t count)` |
| list chunk | `chelis_list *chelis_list_chunk(const chelis_list *list, int64_t size)` |
| integer range | `chelis_list *chelis_range_i64(int64_t start, int64_t end)` |
| list enumerate | `chelis_list *chelis_list_enumerate(const chelis_list *list)` |
| tuple length | `int64_t chelis_tuple_len(const chelis_tuple *tuple)` |
| tuple construction | `chelis_tuple *chelis_tuple_from_values(const chelis_value *items, int64_t len)` |
| tuple index | `chelis_value chelis_tuple_get(const chelis_tuple *tuple, int64_t index)` |
| ADT construction | `chelis_adt *chelis_adt_construct(chelis_string ctor, const chelis_value *fields, int64_t len)` |
| ADT field count | `int64_t chelis_adt_field_count(const chelis_adt *adt)` |
| ADT field index | `chelis_value chelis_adt_get_field(const chelis_adt *adt, int64_t index)` |
| dictionary length | `int64_t chelis_dict_len(const chelis_dict *dict)` |
| dictionary construction | `chelis_dict *chelis_dict_from_pairs(const chelis_list *pairs)` |
| dictionary membership | `bool chelis_dict_contains(const chelis_dict *dict, chelis_value key)` |
| dictionary lookup | `chelis_option *chelis_dict_get(const chelis_dict *dict, chelis_value key)` |
| dictionary removal | `chelis_dict *chelis_dict_remove(const chelis_dict *dict, chelis_value key)` |
| dictionary insertion | `chelis_dict *chelis_dict_insert(const chelis_dict *dict, chelis_value key, chelis_value value)` |
| dictionary merge | `chelis_dict *chelis_dict_merge(const chelis_dict *left, const chelis_dict *right)` |
| byte-file read | `chelis_list *chelis_read_bytes(chelis_string path)` |
| mapped byte read | `chelis_list *chelis_mmap_read(const chelis_mapped_file *mapped, int64_t offset, int64_t len)` |
| mapped byte length | `int64_t chelis_mmap_len(const chelis_mapped_file *mapped)` |
| list print | `void chelis_print_list(const chelis_list *list)` |
| tuple print | `void chelis_print_tuple(const chelis_tuple *tuple)` |
| dictionary print | `void chelis_print_dict(const chelis_dict *dict)` |
| ADT print | `void chelis_print_adt(const chelis_adt *adt)` |""".splitlines()
    ),
    "05-OP-33": tuple(
        """\
| owned allocation | `chelis_tensor *chelis_alloc(int32_t rank, const int64_t *shape, chelis_dtype dtype)` |
| owned allocation with input shape | `chelis_tensor *chelis_tensor_alloc_like(const chelis_tensor *input, chelis_scalar exemplar)` |
| rank | `int32_t chelis_tensor_rank(const chelis_tensor *tensor)` |
| extent | `int64_t chelis_tensor_shape(const chelis_tensor *tensor, int32_t axis)` |
| element count | `int64_t chelis_tensor_numel(const chelis_tensor *tensor)` |
| contiguous metadata plan | `chelis_metadata_plan *chelis_metadata_plan_new(chelis_scalar rank, const chelis_scalar *shape, chelis_scalar exemplar)` |
| strided metadata plan | `chelis_metadata_plan *chelis_metadata_plan_view(chelis_scalar rank, const chelis_scalar *shape, const chelis_scalar *strides, chelis_scalar exemplar, chelis_scalar byte_capacity)` |
| metadata plan rank | `int32_t chelis_metadata_plan_rank(const chelis_metadata_plan *plan)` |
| metadata plan shape | `const int64_t *chelis_metadata_plan_shape(const chelis_metadata_plan *plan)` |
| metadata plan strides | `const int64_t *chelis_metadata_plan_strides(const chelis_metadata_plan *plan)` |
| metadata plan count | `int64_t chelis_metadata_plan_count(const chelis_metadata_plan *plan)` |
| metadata plan logical bytes | `int64_t chelis_metadata_plan_byte_count(const chelis_metadata_plan *plan)` |
| metadata plan dtype | `chelis_dtype chelis_metadata_plan_dtype(const chelis_metadata_plan *plan)` |
| metadata plan capacity check | `void chelis_metadata_plan_check_capacity(const chelis_metadata_plan *plan, chelis_scalar byte_capacity)` |
| metadata plan logical byte offset | `int64_t chelis_metadata_plan_byte_offset(const chelis_metadata_plan *plan, chelis_scalar linear_index)` |
| metadata plan release | `void chelis_metadata_plan_release(chelis_metadata_plan *plan)` |
| contiguous stride | `int64_t chelis_tensor_stride(const chelis_tensor *tensor, int32_t axis)` |
| logical byte count | `int64_t chelis_tensor_byte_count(const chelis_tensor *tensor)` |
| tensor iteration index step | `int64_t chelis_tensor_elementwise_index_step(const chelis_tensor *input, const chelis_tensor *domain)` |
| shape iteration index step | `int64_t chelis_tensor_elementwise_index_step_for_shape(const chelis_tensor *input, chelis_scalar rank, const chelis_scalar *shape)` |
| reshape validation | `void chelis_tensor_check_reshape(const chelis_tensor *tensor, chelis_scalar rank, const chelis_scalar *shape)` |
| owned reshape | `chelis_tensor *chelis_tensor_reshape(const chelis_tensor *tensor, const chelis_list *shape)` |
| contiguous copy | `chelis_tensor *chelis_contiguous(const chelis_tensor *tensor)` |
| typed list ingress | `chelis_tensor *chelis_tensor_from_values(const chelis_list *list, chelis_dtype dtype)` |
| row-major element egress | `chelis_list *chelis_tensor_elements(const chelis_tensor *tensor)` |
| inferred-width padding | `chelis_tensor *chelis_pad_sequences(const chelis_list *sequences, chelis_scalar pad_value)` |
| fixed-width padding | `chelis_tensor *chelis_pad_sequences_to(const chelis_list *sequences, int64_t width, chelis_scalar pad_value)` |
| concatenate | `chelis_tensor *chelis_tensor_concat(const chelis_list *parts, int32_t axis)` |
| split | `chelis_list *chelis_tensor_split(const chelis_tensor *tensor, int32_t axis, const chelis_list *sizes)` |
| gather | `chelis_tensor *chelis_tensor_gather(const chelis_tensor *tensor, const chelis_tensor *indices, int32_t axis)` |
| replace scatter | `chelis_tensor *chelis_tensor_scatter_replace(const chelis_tensor *base, const chelis_tensor *indices, const chelis_tensor *updates, int32_t axis)` |
| additive scatter | `chelis_tensor *chelis_tensor_scatter_add(const chelis_tensor *base, const chelis_tensor *indices, const chelis_tensor *updates, int32_t axis)` |
| comparison | `chelis_tensor *chelis_tensor_cmplt(const chelis_tensor *left, const chelis_tensor *right)` |
| selection | `chelis_tensor *chelis_tensor_where(const chelis_tensor *condition, const chelis_tensor *then_tensor, const chelis_tensor *else_tensor)` |
| inclusive prefix sum | `chelis_tensor *chelis_tensor_cumsum(const chelis_tensor *tensor, int32_t axis)` |
| stable sort | `chelis_tuple *chelis_tensor_sort(const chelis_tensor *tensor, int32_t axis)` |
| diagonal | `chelis_tensor *chelis_tensor_diagonal(const chelis_tensor *tensor, int32_t axis1, int32_t axis2)` |
| trace | `chelis_tensor *chelis_tensor_trace(const chelis_tensor *tensor, int32_t axis1, int32_t axis2)` |
| clamp | `chelis_tensor *chelis_tensor_clamp(const chelis_tensor *tensor, const chelis_tensor *lower, const chelis_tensor *upper)` |
| contraction | `chelis_tensor *chelis_tensor_einsum(chelis_string equation, const chelis_tensor *left, const chelis_tensor *right, chelis_dtype accumulator)` |
| checked coordinate decoding | `void chelis_tensor_unravel_index(const chelis_tensor *tensor, chelis_scalar index, chelis_scalar *coordinates)` |
| checked coordinate encoding | `int64_t chelis_tensor_flat_index(const chelis_tensor *tensor, const chelis_scalar *coordinates)` |
| permutation target validation | `void chelis_tensor_check_permute(const chelis_tensor *tensor, chelis_scalar rank, const chelis_scalar *shape, const chelis_scalar *axes)` |
| expansion target validation | `void chelis_tensor_check_expand(const chelis_tensor *tensor, chelis_scalar rank, const chelis_scalar *shape, int32_t axis)` |
| padding shape construction | `void chelis_tensor_pad_shape(const chelis_tensor *tensor, chelis_scalar rank, const chelis_scalar *before, const chelis_scalar *after, chelis_scalar *shape)` |
| shrinking shape construction | `void chelis_tensor_shrink_shape(const chelis_tensor *tensor, chelis_scalar rank, const chelis_scalar *start, const chelis_scalar *end, chelis_scalar *shape)` |
| striding shape construction | `void chelis_tensor_stride_shape(const chelis_tensor *tensor, chelis_scalar rank, const chelis_scalar *steps, chelis_scalar *shape)` |
| affine coordinate projection | `int64_t chelis_tensor_affine_index(const chelis_tensor *tensor, const chelis_scalar *coordinates, const chelis_scalar *offsets, const chelis_scalar *steps)` |
| checked reduction tensor_reduction_plan | `chelis_reduction_plan *chelis_tensor_reduction_plan(const chelis_tensor *tensor, chelis_scalar axis_count, const chelis_scalar *axes, chelis_scalar exemplar, chelis_reduction_op operation)` |
| checked reduction shape_reduction_plan | `chelis_reduction_plan *chelis_shape_reduction_plan(chelis_scalar rank, const chelis_scalar *shape, chelis_scalar axis_count, const chelis_scalar *axes, chelis_scalar exemplar, chelis_reduction_op operation)` |
| checked reduction reduction_count | `int64_t chelis_reduction_count(const chelis_reduction_plan *plan)` |
| checked reduction reduction_extent | `int64_t chelis_reduction_extent(const chelis_reduction_plan *plan, chelis_scalar axis)` |
| checked reduction reduction_index | `int64_t chelis_reduction_index(const chelis_reduction_plan *plan, chelis_scalar outer, chelis_scalar leaf)` |
| checked reduction reduction_check_target | `void chelis_reduction_check_target(const chelis_reduction_plan *plan, chelis_scalar rank, const chelis_scalar *shape)` |
| checked reduction reduction_check_scratch | `void chelis_reduction_check_scratch(const chelis_reduction_plan *plan, chelis_scalar exemplar)` |
| checked reduction reduction_plan_release | `void chelis_reduction_plan_release(chelis_reduction_plan *plan)` |
| checked sparse tensor_sparse_plan | `chelis_sparse_plan *chelis_tensor_sparse_plan(const chelis_tensor *base, const chelis_tensor *indices, const chelis_tensor *updates, chelis_scalar axis, chelis_sparse_op operation)` |
| checked sparse sparse_extent | `int64_t chelis_sparse_extent(const chelis_sparse_plan *plan, chelis_scalar axis)` |
| checked sparse sparse_count | `int64_t chelis_sparse_count(const chelis_sparse_plan *plan)` |
| checked sparse sparse_index_slot | `int64_t chelis_sparse_index_slot(const chelis_sparse_plan *plan, chelis_scalar linear)` |
| checked sparse sparse_data_index | `int64_t chelis_sparse_data_index(const chelis_sparse_plan *plan, chelis_scalar linear, chelis_scalar selected)` |
| checked sparse sparse_check_target | `void chelis_sparse_check_target(const chelis_sparse_plan *plan, chelis_scalar rank, const chelis_scalar *shape)` |
| checked sparse sparse_plan_release | `void chelis_sparse_plan_release(chelis_sparse_plan *plan)` |
| checked matrix tensor_matmul_plan | `chelis_matmul_plan *chelis_tensor_matmul_plan(const chelis_tensor *left, const chelis_tensor *right, chelis_scalar exemplar)` |
| checked matrix matmul_extent | `int64_t chelis_matmul_extent(const chelis_matmul_plan *plan, chelis_scalar axis)` |
| checked matrix matmul_dimension | `int64_t chelis_matmul_dimension(const chelis_matmul_plan *plan, chelis_matmul_dimension_kind dimension)` |
| checked matrix matmul_batch_count | `int64_t chelis_matmul_batch_count(const chelis_matmul_plan *plan)` |
| checked matrix matmul_matrix_count | `int64_t chelis_matmul_matrix_count(const chelis_matmul_plan *plan, chelis_matmul_part part)` |
| checked matrix matmul_index | `int64_t chelis_matmul_index(const chelis_matmul_plan *plan, chelis_matmul_part part, chelis_scalar batch, chelis_scalar element)` |
| checked matrix matmul_check_target | `void chelis_matmul_check_target(const chelis_matmul_plan *plan, chelis_scalar rank, const chelis_scalar *shape)` |
| checked matrix matmul_check_scratch | `void chelis_matmul_check_scratch(const chelis_matmul_plan *plan, chelis_matmul_part part, chelis_scalar exemplar)` |
| checked matrix matmul_check_vendor | `void chelis_matmul_check_vendor(const chelis_matmul_plan *plan, chelis_scalar maximum)` |
| checked matrix matmul_plan_release | `void chelis_matmul_plan_release(chelis_matmul_plan *plan)` |
| checked movement chelis_tensor_permute_plan | `chelis_movement_plan *chelis_tensor_permute_plan(const chelis_tensor *input, chelis_scalar rank, const chelis_scalar *axes)` |
| checked movement chelis_tensor_expand_plan | `chelis_movement_plan *chelis_tensor_expand_plan(const chelis_tensor *input, chelis_scalar axis, chelis_scalar size, chelis_movement_op operation)` |
| checked movement chelis_tensor_affine_plan | `chelis_movement_plan *chelis_tensor_affine_plan(const chelis_tensor *input, chelis_scalar rank, const chelis_scalar *first, const chelis_scalar *second, chelis_movement_op operation)` |
| checked movement chelis_movement_extent | `int64_t chelis_movement_extent(const chelis_movement_plan *plan, chelis_movement_side side, chelis_scalar axis)` |
| checked movement chelis_movement_count | `int64_t chelis_movement_count(const chelis_movement_plan *plan)` |
| checked movement chelis_movement_index | `int64_t chelis_movement_index(const chelis_movement_plan *plan, chelis_scalar linear)` |
| checked movement chelis_movement_check_target | `void chelis_movement_check_target(const chelis_movement_plan *plan, chelis_scalar rank, const chelis_scalar *shape)` |
| checked movement chelis_movement_plan_release | `void chelis_movement_plan_release(chelis_movement_plan *plan)` |
| checked window tensor_window_plan | `chelis_window_plan *chelis_tensor_window_plan(const chelis_tensor *input, chelis_scalar count, const chelis_scalar *window, const chelis_scalar *steps, chelis_window_op operation)` |
| checked window window_extent | `int64_t chelis_window_extent(const chelis_window_plan *plan, chelis_window_side side, chelis_scalar axis)` |
| checked window window_count | `int64_t chelis_window_count(const chelis_window_plan *plan)` |
| checked window window_index | `int64_t chelis_window_index(const chelis_window_plan *plan, chelis_scalar group, chelis_scalar leaf)` |
| checked window window_check_tensor | `void chelis_window_check_tensor(const chelis_window_plan *plan, const chelis_tensor *tensor, chelis_window_side side)` |
| checked window window_check_target | `void chelis_window_check_target(const chelis_window_plan *plan, chelis_window_side side, chelis_scalar rank, const chelis_scalar *shape)` |
| checked window window_plan_release | `void chelis_window_plan_release(chelis_window_plan *plan)` |
| checked literal tensor_check_literal | `void chelis_tensor_check_literal(chelis_scalar rank, const chelis_scalar *shape, chelis_scalar exemplar, chelis_scalar count)` |
| checked literal tensor_write_literal | `void chelis_tensor_write_literal(chelis_tensor_write *guard, chelis_scalar count, const chelis_scalar *values)` |
| device allocation | `chelis_device_tensor_owner *chelis_device_tensor_alloc(chelis_metadata_plan *plan)` |
| device storage borrow | `chelis_device_tensor_owner *chelis_device_tensor_borrow(chelis_metadata_plan *plan, void *data, chelis_scalar byte_capacity)` |
| device packet import | `chelis_device_tensor_owner *chelis_device_tensor_import(const chelis_gpu_tensor *packet)` |
| device packet observation | `const chelis_gpu_tensor *chelis_device_tensor_view(const chelis_device_tensor_owner *owner)` |
| independent device clone | `chelis_device_tensor_owner *chelis_device_tensor_clone(const chelis_device_tensor_owner *source)` |
| device owner finalization | `void chelis_device_tensor_release(chelis_device_tensor_owner *owner)` |
| host to device transfer | `void chelis_device_tensor_copy_from_host(chelis_device_tensor_owner *destination, const chelis_tensor *source)` |
| device to host transfer | `void chelis_device_tensor_copy_to_host(chelis_tensor_write *destination, const chelis_device_tensor_owner *source)` |
| device owner device | `int32_t chelis_device_tensor_device(const chelis_device_tensor_owner *owner)` |""".splitlines()
    ),
    "05-OP-34": tuple(
        """\
| `io/json::Json` | `JsonNull | JsonBool(bool) | JsonInt(i64) | JsonBigInt(string) | JsonFloat(f64) | JsonString(string) | JsonArray(List[Json]) | JsonObject(Dict[string,Json])` |
| `decimal::Decimal` | `Decimal { coefficient: i64, scale: i64 }` |
| `time::Date` | `Date { year: i64, month: i64, day: i64 }` |
| `time::Duration` | `Duration { days: i64, hours: i64, minutes: i64, seconds: i64 }` |
| `tokenizer::Tokenizer` | `BpeTokenizer(Dict[string,i64], Dict[string,i64], Dict[i64,string], i64)` |""".splitlines()
    ),
    "05-OP-35": tuple(
        """\
| `contracts::normal_cdf` | `(p_float)->p_float` |
| `contracts::normal_cdf_contract_samples` | `()->i64` |
| `contracts::normal_cdf_contract_seed` | `()->i64` |
| `contracts::standard_contract_tolerance` | `()->f32` |
| `decimal::decimal` | `(string)->Decimal` |
| `decimal::decimal_add` | `(Decimal,Decimal)->Decimal` |
| `decimal::decimal_div` | `(Decimal,Decimal,i64,RoundingMode)->Decimal` |
| `decimal::decimal_eq` | `(Decimal,Decimal)->bool` |
| `decimal::decimal_from_int` | `(i64)->Decimal` |
| `decimal::decimal_gt` | `(Decimal,Decimal)->bool` |
| `decimal::decimal_gte` | `(Decimal,Decimal)->bool` |
| `decimal::decimal_lt` | `(Decimal,Decimal)->bool` |
| `decimal::decimal_lte` | `(Decimal,Decimal)->bool` |
| `decimal::decimal_mul` | `(Decimal,Decimal)->Decimal` |
| `decimal::decimal_sub` | `(Decimal,Decimal)->Decimal` |
| `decimal::decimal_to_float` | `(Decimal)->f64` |
| `decimal::decimal_to_string` | `(Decimal)->string` |
| `decimal::try_decimal` | `(string)->Option[Decimal]` |
| `index::list_index` | `(List[T],i64)->T` |
| `index::skip_list` | `(List[T],i64)->List[T]` |
| `index::take_list` | `(List[T],i64)->List[T]` |
| `init/kaiming::kaiming_normal` | `(key,&tensor[..r,p_float],p_float)->tensor[..r,p_float]` |
| `init/kaiming::kaiming_uniform` | `(key,&tensor[..r,p_float],p_float)->tensor[..r,p_float]` |
| `init/random::normal_like` | `(key,&tensor[..r,p_float],p_float,p_float)->tensor[..r,p_float]` |
| `init/xavierext::trunc_normal` | `(key,&tensor[..r,p_float],p_float,p_float,p_float,p_float)->tensor[..r,p_float]` |
| `init/xavierext::xavier_normal` | `(key,&tensor[..r,p_float],p_float,p_float)->tensor[..r,p_float]` |
| `init/xavierext::xavier_uniform` | `(key,&tensor[..r,p_float],p_float,p_float)->tensor[..r,p_float]` |
| `io/json::json_array` | `(Option[Json])->Option[List[Json]]` |
| `io/json::json_bigint` | `(Option[Json])->Option[string]` |
| `io/json::json_bool` | `(Option[Json])->Option[bool]` |
| `io/json::json_float` | `(Option[Json])->Option[f64]` |
| `io/json::json_get` | `(Json,string)->Option[Json]` |
| `io/json::json_int` | `(Option[Json])->Option[i64]` |
| `io/json::json_is_null` | `(Option[Json])->bool` |
| `io/json::json_object` | `(Option[Json])->Option[Dict[string,Json]]` |
| `io/json::json_string` | `(Option[Json])->Option[string]` |
| `io/json::load_json` | `(string)->Json!{IO}` |
| `io/json::parse_json` | `(string)->Json` |
| `io/json::to_json` | `(Json)->string` |
| `io/json::try_load_json` | `(string)->Option[Json]!{IO}` |
| `io/json::try_parse_json` | `(string)->Option[Json]` |
| `io/json::try_to_json` | `(Json)->Option[string]` |
| `io/json::try_write_json` | `(string,Json)->Option[unit]!{IO}` |
| `io/json::write_json` | `(string,Json)->unit!{IO}` |
| `io::mmap_size` | `(string)->i64!{IO}` |
| `io::read_head_bytes` | `(string,i64)->List[i64]!{IO}` |
| `process::run` | `(string,List[string])->(i64,string,string)!{IO}` |
| `process::run_chelis` | `(List[string])->(i64,string,string)!{IO}` |
| `scalar::abs` | `(p_numeric)->p_numeric` |
| `scalar::max` | `(p_numeric,p_numeric)->p_numeric` |
| `scalar::min` | `(p_numeric,p_numeric)->p_numeric` |
| `sort::sort` | `(&tensor[..r,p_numeric],i32)->(tensor[..r,p_numeric],tensor[..r,i64])` |
| `tensor/construct::arange` | `(p_int,p_int)->tensor[n,p_int]` |
| `tensor/construct::linspace` | `(p_float,p_float,i64)->tensor[n,p_float]` |
| `tensor/construct::squeeze` | `(&tensor[..pre,1,..post,p],i32)->tensor[..pre,..post,p]` |
| `tensor/construct::stack` | `(List[tensor[..pre,..post,p]],i32)->tensor[..pre,rows,..post,p]` |
| `tensor/construct::unsqueeze` | `(&tensor[..pre,..post,p],i32)->tensor[..pre,1,..post,p]` |
| `tensor/mask::where_indices` | `(&tensor[..r,bool])->tensor[hits,i64]` |
| `test::assert_close` | `(p_float,p_float,p_float,string)->unit!{Test}` |
| `test::assert_close_tensor` | `(&tensor[..r,p_float],&tensor[..r,p_float],p_float,string)->unit!{Test}` |
| `test::assert_eq` | `(Q,Q,string)->unit!{Test}` |
| `test::assert_eq_tensor` | `(&tensor[..r,p],&tensor[..r,p],string)->unit!{Test}` |
| `test::assert_shape` | `(&tensor[..r,p],List[i64],string)->unit!{Test}` |
| `time::add_days` | `(Date,i64)->Date` |
| `time::date` | `(i64,i64,i64)->Date` |
| `time::date_gt` | `(Date,Date)->bool` |
| `time::date_gte` | `(Date,Date)->bool` |
| `time::date_lt` | `(Date,Date)->bool` |
| `time::date_lte` | `(Date,Date)->bool` |
| `time::date_to_string` | `(Date)->string` |
| `time::day_of_week` | `(Date)->DayOfWeek` |
| `time::day_of_week_name` | `(Date)->string` |
| `time::day_of_year` | `(Date)->i64` |
| `time::days_between` | `(Date,Date)->i64` |
| `time::duration` | `(i64,i64,i64,i64)->Duration` |
| `time::is_leap_year` | `(i64)->bool` |
| `time::parse_date` | `(string)->Option[Date]` |
| `time::sub_days` | `(Date,i64)->Date` |
| `time::try_date` | `(i64,i64,i64)->Option[Date]` |
| `tokenizer::batch_encode` | `(Tokenizer,List[string],i64,i64)->tensor[batch,seq,i64]` |
| `tokenizer::decode` | `(Tokenizer,List[i64])->string` |
| `tokenizer::encode` | `(Tokenizer,string)->List[i64]` |
| `tokenizer::load_tokenizer` | `(string)->Tokenizer!{IO}` |
| `tokenizer::try_load_tokenizer` | `(string)->Option[Tokenizer]!{IO}` |""".splitlines()
    ),
    "05-OP-44": tuple(
        """\
| string retain | `void chelis_string_retain(chelis_string value)` |
| string release | `void chelis_string_release(chelis_string value)` |
| tensor retain | `void chelis_tensor_retain(const chelis_tensor *tensor)` |
| tensor release | `void chelis_tensor_release(const chelis_tensor *tensor)` |
| list retain | `void chelis_list_retain(const chelis_list *list)` |
| list release | `void chelis_list_release(const chelis_list *list)` |
| tuple retain | `void chelis_tuple_retain(const chelis_tuple *tuple)` |
| tuple release | `void chelis_tuple_release(const chelis_tuple *tuple)` |
| dictionary retain | `void chelis_dict_retain(const chelis_dict *dict)` |
| dictionary release | `void chelis_dict_release(const chelis_dict *dict)` |
| ADT retain | `void chelis_adt_retain(const chelis_adt *adt)` |
| ADT release | `void chelis_adt_release(const chelis_adt *adt)` |
| option retain | `void chelis_option_retain(const chelis_option *option)` |
| option release | `void chelis_option_release(const chelis_option *option)` |
| mapped-file retain | `void chelis_mapped_file_retain(const chelis_mapped_file *mapped)` |
| mapped-file release | `void chelis_mapped_file_release(const chelis_mapped_file *mapped)` |
| value clone | `chelis_value chelis_value_clone(chelis_value value)` |
| value release | `void chelis_value_release(chelis_value value)` |
| string take into value | `chelis_value chelis_value_take_string(chelis_string value)` |
| tensor take into value | `chelis_value chelis_value_take_tensor(chelis_tensor *tensor)` |
| list take into value | `chelis_value chelis_value_take_list(chelis_list *list)` |
| tuple take into value | `chelis_value chelis_value_take_tuple(chelis_tuple *tuple)` |
| dictionary take into value | `chelis_value chelis_value_take_dict(chelis_dict *dict)` |
| ADT take into value | `chelis_value chelis_value_take_adt(chelis_adt *adt)` |
| option take into value | `chelis_value chelis_value_take_option(chelis_option *option)` |
| mapped-file take into value | `chelis_value chelis_value_take_mapped_file(chelis_mapped_file *mapped)` |
| string take out of value | `chelis_string chelis_string_take_value(chelis_value value)` |
| string borrow from value | `chelis_string chelis_string_borrow_value(chelis_value value)` |
| tensor take out of value | `chelis_tensor *chelis_tensor_take_value(chelis_value value)` |
| tensor borrow from value | `const chelis_tensor *chelis_tensor_borrow_value(chelis_value value)` |
| list take out of value | `chelis_list *chelis_list_take_value(chelis_value value)` |
| list borrow from value | `const chelis_list *chelis_list_borrow_value(chelis_value value)` |
| tuple take out of value | `chelis_tuple *chelis_tuple_take_value(chelis_value value)` |
| tuple borrow from value | `const chelis_tuple *chelis_tuple_borrow_value(chelis_value value)` |
| dictionary take out of value | `chelis_dict *chelis_dict_take_value(chelis_value value)` |
| dictionary borrow from value | `const chelis_dict *chelis_dict_borrow_value(chelis_value value)` |
| ADT take out of value | `chelis_adt *chelis_adt_take_value(chelis_value value)` |
| ADT borrow from value | `const chelis_adt *chelis_adt_borrow_value(chelis_value value)` |
| option take out of value | `chelis_option *chelis_option_take_value(chelis_value value)` |
| option borrow from value | `const chelis_option *chelis_option_borrow_value(chelis_value value)` |
| mapped-file take out of value | `chelis_mapped_file *chelis_mapped_file_take_value(chelis_value value)` |
| mapped-file borrow from value | `const chelis_mapped_file *chelis_mapped_file_borrow_value(chelis_value value)` |
| option none | `chelis_option *chelis_option_none(void)` |
| option some | `chelis_option *chelis_option_some(chelis_value value)` |
| option discriminant | `bool chelis_option_is_some(const chelis_option *option)` |
| option unwrap | `chelis_value chelis_option_unwrap(const chelis_option *option)` |
| tensor entry borrow | `chelis_tensor *chelis_tensor_entry_borrow(int32_t rank, const int64_t *shape, chelis_dtype dtype, const void *data, int64_t byte_capacity)` |
| tensor read view | `chelis_read_view chelis_tensor_read_view(const chelis_tensor *tensor)` |
| tensor begin write | `chelis_tensor_write *chelis_tensor_begin_write(chelis_tensor *tensor)` |
| tensor write view | `chelis_write_view chelis_tensor_write_view(const chelis_tensor_write *guard)` |
| tensor end write | `void chelis_tensor_end_write(chelis_tensor_write *guard)` |
| tensor repurpose | `void chelis_tensor_repurpose(chelis_tensor *tensor, chelis_scalar rank, const chelis_scalar *shape)` |""".splitlines()
    ),
    "05-OP-38": tuple(
        """\
> | `tensor_scan` | `(T,((T,i64)->T!E),i64)->tensor[n,..state_shape(T),element(T)]!E` |
> | `process_run` | `(string,List[string])->(i64,string,string)!{IO}` |
> | `test_assert_eq` | `(Q,Q,string)->unit!{Test}` |
> | `test_assert_close_tensor` | `(&tensor[..r,p_float],&tensor[..r,p_float],p_float,string)->unit!{Test}` |
> | `test_assert_eq_tensor` | `(&tensor[..r,p],&tensor[..r,p],string)->unit!{Test}` |""".splitlines()
    ),
}

REQUIRED_ATOMS = (
    '05-OP-45',
    '04-FIT-18',
    '04-LIN-3',
    '04-LIN-4',
    '04-LIN-5',
    '04-LIN-6',
    '04-LIN-7',
    '04-LIN-8',
    '04-NUM-2',
    '04-NUM-4',
    '04-NUM-8',
    '04-NUM-11',
    '04-NUM-14',
    '04-NUM-16',
    '04-SHAPE-1',
    '05-OP-1',
    '05-OP-2',
    '05-OP-3',
    '05-OP-4',
    '05-OP-5',
    '05-OP-6',
    '05-OP-7',
    '05-OP-8',
    '05-OP-9',
    '05-OP-10',
    '05-OP-11',
    '05-OP-12',
    '05-OP-13',
    '05-OP-14',
    '05-OP-15',
    '05-OP-16',
    '05-OP-17',
    '05-OP-18',
    '05-OP-19',
    '05-OP-20',
    '05-OP-21',
    '05-OP-22',
    '05-OP-23',
    '05-OP-24',
    '05-OP-25',
    '05-OP-26',
    '05-OP-27',
    '05-OP-28',
    '05-OP-29',
    '05-OP-30',
    '05-OP-31',
    '05-OP-32',
    '05-OP-33',
    '05-OP-34',
    '05-OP-35',
    '05-OP-36',
    '05-OP-37',
    '05-OP-38',
    '05-OP-39',
    '05-OP-40',
    '05-OP-41',
    '05-OP-42',
    '05-OP-43',
    '05-OP-44',
)

# Each boundary must occur exactly once; the end marker is excluded from the
# reported region. An intentional declaration change owes its owning contract,
# consumers, adversarial controls and exact PR acknowledgement.
REQUIRED_REGIONS = {
    'Python numeric boundary': (
        'spec/11-ffi.md',
        '## 1. Python Interop',
        '## 2. C Interop',
    ),
    'numeric wire codecs and roles': (
        'spec/10-serialization.md',
        '### 3.2 Exact Numeric Value Codec',
        '## 4. Invariant Revalidation At Decode Boundaries',
    ),
    'numeric value semantics': (
        'spec/04-type-system.md',
        '## 9. Numeric Value Semantics',
        '## 10. Checker Totality',
    ),
    'numeric primitive contracts': (
        'spec/05-risc-primitives.md',
        '### 2.1 Elementwise Binary',
        '### 2.4 Movement',
    ),
    'logical builtin contract': (
        'spec/05-risc-primitives.md',
        '### 3.2 Comparison and Logical Operations',
        '### 3.3 Activation Functions',
    ),
    'name-preserving rank polymorphism': (
        'spec/04-type-system.md',
        '#### 4.5.3 Name-Preserving Rank Polymorphism',
        '#### 4.5.4 Concat Result Typing',
    ),
    'window extrema contract': (
        'spec/05-risc-primitives.md',
        '### 2.3.1 Windowed Reduction',
        '### 2.4 Movement',
    ),
    'to_string contract section': (
        'spec/05-risc-primitives.md',
        '### 3.6.3 Canonical value-to-string conversion',
        '### 3.7 Host-Lane Data I/O Numeric Operations',
    ),
    'named lossy cast section': (
        'spec/05-risc-primitives.md',
        '### 3.8 Named Lossy Cast Forms',
        '## 4. Standard Lowerings',
    ),
    'capability schema': (
        'spec/design/capability_table.md',
        '## The two-table design',
        '## Seed dispositions the table must ship with',
    ),
    'capability seed dispositions': (
        'spec/design/capability_table.md',
        '## Seed dispositions the table must ship with',
        '## New numeric ops before the table lands (added 2026-07-30)',
    ),
    'Phase 4 handoff': (
        'spec/design/dtype_semantics.md',
        '## Phase 4 - the capability table becomes the permanent guard',
        '## I1. Interlock with loud unsupported ([#730])',
    ),
    'compiled stdlib consumer': (
        'spec/design/loud_unsupported.md',
        '### LU5 - derived compiled-stdlib acceptance corpus ([#955])',
        '### LU6 - exhaustive checked host-cast planning ([#1150])',
    ),
    'provenance governed surfaces': (
        'spec/design/spec_provenance.md',
        '| Capability-row, Deep-tag, tolerance-row, and diagnostic citations |',
        '| OpenSpec proving inadequate as the trigger for provenance work |',
    ),
    'roadmap ownership': (
        'spec/design/remediation_roadmap.md',
        '| **v0.19.0 - grounded dtype storage break',
        '| **v0.20.0 - behavior-preserving permanent guards**',
    ),
    'status dtype row': (
        'docs/investigations/remediation_status_2026_08_04.md',
        '| **#729 dtype semantics** |',
        '| **#730 loud unsupported** |',
    ),
}


def read(root: Path, relative: str) -> str:
    path = root / relative
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise OracleError(f"cannot read {relative}: {error}") from error


def require(text: str, fragment: str, label: str, violations: list[str]) -> None:
    if fragment not in text:
        violations.append(f"missing {label}")


def require_all(
    text: str,
    requirements: tuple[tuple[str, str], ...],
    violations: list[str],
) -> None:
    for fragment, label in requirements:
        require(text, fragment, label, violations)


def atom_blocks(text: str) -> dict[str, str]:
    matches = list(ATOM_START.finditer(text))
    blocks: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        blocks[match.group(1)] = text[match.start() : end]
    return blocks


OP_MANIFEST_REGISTRY_FILES = {
    "05-OP-45": "spec/registry/python_tensor_metadata.md",
    "05-OP-31": "spec/registry/c_scalar_carrier.md",
    "05-OP-32": "spec/registry/c_container_boundary.md",
    "05-OP-33": "spec/registry/c_tensor_runtime.md",
    "05-OP-44": "spec/registry/c_heap_lifetime.md",
    "05-OP-34": "spec/registry/stdlib_adt_identities.md",
    "05-OP-35": "spec/registry/stdlib_numeric_manifest.md",
}


def manifest_rows(registry_text: str) -> tuple[str, ...]:
    """Return normative registry data rows, excluding headers/separators."""

    return tuple(
        line
        for line in registry_text.splitlines()
        if line.startswith("| ")
        and "`" in line
        and not re.match(r"^\| (callable|identity) \|", line)
    )


def block_manifest_rows(block: str) -> tuple[str, ...]:
    """Return in-atom Markdown data rows for atoms that keep their table."""

    return tuple(
        line
        for line in block.splitlines()
        if line.startswith("> | ") and "`" in line
    )


def validate_op_manifests(
    docs: dict[str, str], blocks: dict[str, str], violations: list[str]
) -> None:
    for atom, expected in EXPECTED_OP_MANIFESTS.items():
        block = blocks.get(atom, "")
        relative = OP_MANIFEST_REGISTRY_FILES.get(atom)
        if relative is None:
            actual = block_manifest_rows(block)
        else:
            registry_text = docs.get(relative, "")
            actual = manifest_rows(registry_text)
            if f"`{relative}`" not in block:
                violations.append(
                    f"[{atom}] must incorporate its normative registry "
                    f"{relative} by reference"
                )
            if re.search(r"^> \| ", block, re.MULTILINE):
                violations.append(
                    f"[{atom}] may not carry a second manifest copy; rows live "
                    f"only in {relative}"
                )
            if f"[{atom}]" not in registry_text:
                violations.append(
                    f"registry {relative} must name its owning atom [{atom}]"
                )
        if actual == expected:
            continue
        missing = [row for row in expected if row not in actual]
        extra = [row for row in actual if row not in expected]
        detail: list[str] = []
        if missing:
            detail.append(f"missing {missing[0]}")
        if extra:
            detail.append(f"unexpected {extra[0]}")
        if not detail:
            detail.append("canonical row order changed")
        violations.append(f"[{atom}] exact manifest mismatch: {'; '.join(detail)}")

    stdlib_rows = re.findall(
        r"^\| `([^`]+)` \| `([^`]+)` \|$",
        docs.get(OP_MANIFEST_REGISTRY_FILES["05-OP-35"], ""),
        re.MULTILINE,
    )
    identities = [identity for identity, _signature in stdlib_rows]
    if len(identities) != 84 or len(set(identities)) != 84:
        violations.append(
            "[05-OP-35] stdlib numeric manifest must have exactly eighty-four "
            "unique identities"
        )


# --------------------------------------------------------------------------
# Frozen-contract acknowledgement gate (the additive-contradiction leg)
# --------------------------------------------------------------------------

DEFAULT_BASE_REF = "origin/main"

def _git(root: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ("git", "-C", str(root), *args),
        capture_output=True,
        check=False,
    )


def resolve_merge_base(root: Path, base: str) -> str:
    """Return the merge-base commit of ``base`` and ``HEAD``.

    Raises ``OracleError`` when it cannot be determined. There is no "assume
    unchanged" path: a base this function cannot resolve is a check that did
    not run.
    """

    inside = _git(root, "rev-parse", "--is-inside-work-tree")
    if inside.returncode != 0 or inside.stdout.strip() != b"true":
        raise OracleError(
            f"cannot determine the frozen contract merge base: {root} is not a "
            "git work tree"
        )
    merge_base = _git(root, "merge-base", base, "HEAD")
    if merge_base.returncode != 0:
        detail = merge_base.stderr.decode("utf-8", "replace").strip()
        raise OracleError(
            f"cannot determine the frozen contract merge base with {base!r}"
            + (f": {detail}" if detail else "")
            + "; fetch the base branch or pass --base <ref>"
        )
    return merge_base.stdout.decode("utf-8").strip()


def contract_files_at(
    root: Path, merge_base: str, contract_files: tuple[str, ...]
) -> set[str]:
    """Return the subset of ``contract_files`` present at ``merge_base``.

    Absence and an unreadable object store are different failures, and reading
    the second as the first would report a changed file as unchanged. This
    enumerates the tree once so a later `git show` failure is an error.
    """

    listing = _git(
        root, "ls-tree", "-r", "-z", "--name-only", merge_base, "--", *contract_files
    )
    if listing.returncode != 0:
        detail = listing.stderr.decode("utf-8", "replace").strip()
        raise OracleError(
            f"cannot list frozen contract files at {merge_base}"
            + (f": {detail}" if detail else "")
        )
    present = {
        name
        for name in listing.stdout.decode("utf-8", "surrogateescape").split("\0")
        if name
    }
    return present & set(contract_files)


def changed_contract_files(
    root: Path, merge_base: str, contract_files: tuple[str, ...] = CONTRACT_FILES
) -> list[str]:
    """Return the contract files whose bytes differ from ``merge_base``."""

    present = contract_files_at(root, merge_base, contract_files)
    changed: list[str] = []
    for relative in contract_files:
        baseline: bytes | None = None
        if relative in present:
            blob = _git(root, "show", f"{merge_base}:{relative}")
            if blob.returncode != 0:
                detail = blob.stderr.decode("utf-8", "replace").strip()
                raise OracleError(
                    f"cannot read {relative} at {merge_base}"
                    + (f": {detail}" if detail else "")
                )
            baseline = blob.stdout
        path = root / relative
        try:
            current: bytes | None = path.read_bytes()
        except OSError:
            current = None
        if baseline != current:
            changed.append(relative)
    return changed


def validate_frozen_contract_changes(
    root: Path = REPO_ROOT,
    base: str = DEFAULT_BASE_REF,
    acknowledgements: tuple[str, ...] = (),
    body: str | None = None,
    require_acknowledgement: bool = False,
    contract_files: tuple[str, ...] = CONTRACT_FILES,
    committed_contract_changes: tuple[str, ...] = (),
) -> list[str]:
    """Check that every changed contract file is acknowledged by name.

    ``committed_contract_changes`` retains file additions/removals named by both
    committed inventories, including a removed declaration whose file remains.
    Returns the report lines. In ``require_acknowledgement`` mode any violation
    raises ``OracleError``; otherwise the report is advisory and the caller
    continues.
    """

    violations: list[str] = []
    acknowledged: list[str] = list(acknowledgements)
    if body is not None:
        parsed, errors = parse_acknowledgements(body)
        acknowledged.extend(parsed)
        violations.extend(errors)

    duplicates = sorted(
        {path for path, count in Counter(acknowledged).items() if count > 1}
    )
    for path in duplicates:
        violations.append(
            f"duplicate frozen contract acknowledgement for {path}: acknowledge "
            "each changed contract file exactly once"
        )

    try:
        merge_base = resolve_merge_base(root, base)
        changed = sorted(set(changed_contract_files(root, merge_base, contract_files))
                         | set(committed_contract_changes))
    except OracleError as error:
        # Advisory mode reports and continues so the independent contract checks
        # still run on a checkout whose git state this leg cannot read. The
        # enforcing mode re-raises: a check that did not run is never a pass.
        if require_acknowledgement:
            raise
        return [
            f"frozen contract acknowledgement: {error}",
            "frozen contract acknowledgement: change detection skipped "
            "(advisory mode); the dedicated PR acknowledgement check fails "
            "on an unreadable base",
        ]

    changed_set = set(changed)
    known = set(contract_files) | set(committed_contract_changes)

    for path in sorted(set(acknowledged)):
        if path not in known:
            violations.append(
                f"frozen contract acknowledgement names {path}, which is not a "
                "frozen contract file"
            )
        elif path not in changed_set:
            violations.append(
                f"stale frozen contract acknowledgement for {path}: it is "
                f"unchanged against {base} ({merge_base[:12]}); remove the "
                f"'{ACKNOWLEDGEMENT_KEY} {path}' line"
            )

    acknowledged_set = set(acknowledged)
    for path in changed:
        if path not in acknowledged_set:
            violations.append(
                f"unacknowledged frozen contract change: {path} differs from "
                f"{base} ({merge_base[:12]}); add the line "
                f"'{ACKNOWLEDGEMENT_KEY} {path}' to the pull request body, and "
                "with it the owning spec/design update, every consuming "
                "contract, and an adversarial mutation"
            )

    if violations and require_acknowledgement:
        raise OracleError("; ".join(violations))

    report = [
        f"frozen contract acknowledgement: base {base} ({merge_base[:12]}), "
        f"{len(changed)} of {len(contract_files)} contract files changed"
    ]
    for path in changed:
        marker = "ok " if path in acknowledged_set else "NEEDS"
        report.append(f"  {marker} {ACKNOWLEDGEMENT_KEY} {path}")
    report.extend(f"  ISSUE {violation}" for violation in violations)
    return report


def validate_required_contract(
    docs: dict[str, str], violations: list[str]
) -> None:
    # Required identities and boundaries survive text-hash retirement. The
    # committed report and enforcing acknowledgements own deliberate text edits;
    # required semantic clauses below still fail independently of Git history.
    for atom in REQUIRED_ATOMS:
        relative = (
            "spec/04-type-system.md" if atom.startswith("04-") else "spec/05-risc-primitives.md"
        )
        try:
            strict_atom_block(docs[relative], atom)
        except OracleError as error:
            violations.append(str(error))
    for label, (relative, start, end) in REQUIRED_REGIONS.items():
        try:
            frozen_region(docs[relative], start, end, label)
        except OracleError as error:
            violations.append(str(error))


def normalize_atom_body(body: str) -> str:
    normalized = " ".join(
        line[2:] if line.startswith("> ") else line
        for line in body.splitlines()
    )
    return " ".join(normalized.split())


def require_atom(
    blocks: dict[str, str],
    atom: str,
    requirements: tuple[str, ...],
    violations: list[str],
) -> None:
    body = blocks.get(atom)
    if body is None:
        violations.append(f"missing normative atom [{atom}]")
        return
    normalized_body = normalize_atom_body(body)
    for fragment in requirements:
        normalized_fragment = " ".join(
            line[2:] if line.startswith("> ") else line
            for line in fragment.splitlines()
        )
        normalized_fragment = " ".join(normalized_fragment.split())
        if normalized_fragment not in normalized_body:
            violations.append(f"[{atom}] missing semantic clause: {fragment}")


def validate_normative_contract(
    docs: dict[str, str], violations: list[str]
) -> None:
    spec02 = docs["spec/02-surf-syntax.md"]
    spec03 = docs["spec/03-deep-syntax.md"]
    spec04 = docs["spec/04-type-system.md"]
    require(
        spec04,
        "`and` / `or` / `not` are the\n  logical operations and "
        "`count` is the bool-tensor counting operation",
        "bool counting operation",
        violations,
    )
    spec05 = docs["spec/05-risc-primitives.md"]
    spec06 = docs["spec/06-transformations.md"]
    spec10 = docs["spec/10-serialization.md"]
    spec11 = docs["spec/11-ffi.md"]
    ownership_design = docs["spec/design/compiled_value_ownership.md"]
    implicit_linearity = docs["spec/design/implicit_linearity.md"]

    atoms = [int(number) for number in OP_ATOM.findall(spec05)]
    counts = Counter(atoms)
    duplicates = sorted(number for number, count in counts.items() if count != 1)
    if duplicates:
        violations.append(f"duplicate normative OP atoms: {duplicates}")
    blocks = atom_blocks(spec05)
    for number, expected_heading in EXPECTED_PHASE4B_OP_HEADINGS.items():
        atom = f"05-OP-{number}"
        body = blocks.get(atom)
        normalized = normalize_atom_body(body) if body is not None else ""
        expected_prefix = f"**[{atom}]** {expected_heading}"
        if not normalized.startswith(expected_prefix):
            violations.append(
                f"[{atom}] must begin `{expected_heading}`, got "
                f"{normalized.splitlines()[0] if normalized else 'missing'}"
            )
    if re.search(r"^> \*\*\[05-OP-\d+\]\*\* `cast_round\b", spec05, re.MULTILINE):
        violations.append("cast_round must not be a normative operation atom")

    require_all(
        spec02,
        (
            ("Surf has no infinity or NaN literal.", "Surf literal exclusion"),
            (
                "`count`\nlowers once with its complete named-axis vector",
                "Surf count multi-axis lowering",
            ),
            (
                "DtypeFamily   <- 'Float' / 'Int' / 'Numeric'",
                "Surf dtype-family bound production",
            ),
            (
                "A `sig`'s `[..]` clause is complete: every `t-var`, `d-var`, and\n"
                "`d-rank` name in the signature appears exactly once.",
                "Surf complete declaration binder list",
            ),
            (
                "A listed name **that declares\na bound** must occur in the declared type.",
                "Surf occurrence rule is bounded-binder only",
            ),
            (
                "One binder list owns each\ndeclaration: a standalone `sig` carries it, "
                "and a matching `def` must\nnot carry a second list.",
                "Surf single declaration binder-list owner",
            ),
        ),
        violations,
    )
    require_all(
        spec03,
        (
            (
                "Canonical Deep contains no non-finite float literal.",
                "Deep literal exclusion",
            ),
            ("**Reduce:** `sum`, `count`, `max_reduce`", "Deep count builtin"),
            (
                "| `dtype_bounds` | metadata map | Dtype-family bounds on a "
                "`defsig`'s binders; see §2.2 |",
                "Deep dtype-family bound metadata key",
            ),
            (
                "MetaKey     \u2190 [A-Za-z_] [A-Za-z0-9_]*",
                "Deep grammar derives the declared metadata key charset",
            ),
            (
                "MetaValue   \u2190 Meta / Node / Literal / Identifier / TypeName",
                "Deep grammar derives a map-valued metadata key",
            ),
        ),
        violations,
    )
    require_all(
        spec04,
        (
            ("The active primitive set is exactly eleven names", "eleven active primitives"),
            (
                "one of the eleven active primitives is well-typed",
                "backend-neutral active primitive set",
            ),
            ("ten active tensor element dtypes", "ten tensor element dtypes"),
            ("nine active data element dtypes", "nine data element dtypes"),
            (
                "For an unconsumed local owner, the compiler inserts `Drop` at the "
                "earliest\npost-dominating point after its last use",
                "linearity last-use Drop placement",
            ),
            (
                "Lexical scope\nexit is the fallback only when no earlier valid "
                "terminal point can be proved",
                "linearity scope-exit fallback",
            ),
            (
                "Unifying two bounded variables SHALL\n> yield the intersection "
                "of their families.",
                "dtype-family bound intersection",
            ),
            (
                "an empty intersection SHALL be a `PrecisionMismatch` naming both\n"
                "> families.",
                "empty intersection names both families",
            ),
            (
                "A binder that declares no bound\n> remains an unconstrained type "
                "variable admitting every type, not only a\n> dtype.",
                "unbounded binder stays a general type variable",
            ),
            (
                "| `Numeric` | the union of `Float` and `Int` |",
                "dtype-family membership table",
            ),
            (
                "A public stdlib signature whose `[05-OP-35]` registry domain "
                "is exactly one of\nthese families declares that family as a bound.",
                "stdlib bound obligation cites the registry domain",
            ),
        ),
        violations,
    )
    require_all(
        spec05,
        (
            (
                "Source borrow syntax and primitive-DAG borrow markers\n"
                "are erased before backend emission",
                "primitive source-marker erasure",
            ),
            (
                "The resolved disposition of every use is not erased; ownership\n"
                "lowering first records explicit borrow, move, clone, and terminal "
                "`Drop` obligations\nin the verified ownership representation "
                "consumed by every backend",
                "primitive verified ownership preservation",
            ),
        ),
        violations,
    )
    require_all(
        spec10,
        (
            ("Schema version 21 is explicitly\npresent", "wire v21 presence"),
            ("versions 1 through 20", "wire old-version rejection"),
            (
                "and every future version are decode errors before any IR\n"
                "node is consumed",
                "wire future-version rejection",
            ),
            ("the only accepted version", "wire current-version exactness"),
            ("There is no versionless default", "wire versionless rejection"),
            ("versionless default, legacy migration", "wire migration rejection"),
            (
                "Every requirement uses the exact\n`NonnegativeExtent` adapter over a nonnegative `int64`",
                "wire literal-witness requirement carrier",
            ),
            (
                "`WireDagNode.shape_deps` contains exact u64 node\nreferences to strictly earlier nodes",
                "wire shape-dependency references",
            ),
            (
                "`shape_deps`, `span_id` (explicitly null when absent), `merged_spans` and\n`declaration` are mandatory fields",
                "wire mandatory invocation fields",
            ),
            ("WireRiscOp::Count { axes }", "wire count variant"),
            (
                "complete\nnon-empty vector of unique normalized original-axis "
                "positions in strictly\ndescending order",
                "wire count canonical axes",
            ),
            (
                "decoder rejects an empty,\nduplicate, increasing, or out-of-range "
                "vector before IR construction",
                "wire count axis rejection",
            ),
            ("WireRiscOp::Pad { fill: ScalarValue, ... }", "wire typed Pad variant"),
            (
                "raw JSON number, an untagged payload, a string-mode\nfill, or a "
                "mismatched dtype is a decode error before IR construction",
                "wire Pad payload rejection",
            ),
            (
                "No\nnumeric-fill migration or inferred fill dtype exists",
                "wire Pad no compatibility",
            ),
        ),
        violations,
    )
    require_all(
        spec11,
        (
            ("compiled entry borrows every input runtime value", "FFI entry borrow"),
            ("one owned runtime value for every owned result", "FFI owned result"),
            (
                "never\nreleases or mutates an input's storage",
                "FFI input preservation",
            ),
            ("independently owned and may\nbe released in either order", "FFI root owners"),
            ("governed by [05-OP-31..33]", "FFI complete C authority range"),
        ),
        violations,
    )
    require_all(
        spec10,
        (
            ("f64: 16; f32: 8; f16: 4; bf16: 4", "wire IEEE bit widths"),
            ("No codec normalizes a NaN payload or a signed zero.", "wire bit preservation"),
            ("A raw source DTO is not an admitted executable AST.", "wire raw-source admission"),
            ("A reference is resolved only in its declared owner and namespace.", "wire reference scope"),
            ("Bounds alone never establish transport authority.", "wire report numeric authority"),
            ("`schema_version: 4`", "execution v4 exactness"),
        ),
        violations,
    )
    require_all(
        spec04,
        (("untyped_nodes = total_nodes - typed_nodes", "fitness counter consistency"),),
        violations,
    )
    require_all(
        spec11,
        (
            ("Dynamic Python object types do not establish nonnumeric capacity.", "binding dynamic capacity"),
            ("DLPack keywords are validated, never ignored.", "binding DLPack keyword admission"),
        ),
        violations,
    )
    require_all(
        docs["spec/design/dtype_semantics.md"],
        (("No partial WireDag v9 is published.", "wire atomic cutover"),),
        violations,
    )
    require_all(
        ownership_design,
        (
            (
                "No arrow after verification may accept the pre-verification form",
                "verified backend boundary",
            ),
            (
                "only `verify_ownership` constructs\n`VerifiedOwnershipProgram`",
                "private verification constructor",
            ),
            (
                "MappedFile` is included even though no current [#1286] child "
                "names it",
                "closed mapped-file kind",
            ),
            (
                "one total, wildcard-free ownership classification",
                "closed host/carrier/heap classification",
            ),
            (
                "Every target-representable `Option<T>`, including `Option` of a "
                "scalar, mapped\nresource, or another `Option`",
                "recursive Option heap classification",
            ),
            ("    Option,\n    MappedFile,", "Option heap kind"),
            (
                "| `string` | `String` | fixed `chelis_string` wrapper with an "
                "opaque target and `CHELIS_VALUE_STRING` |",
                "string carrier mapping",
            ),
            (
                "| tensor value or internal tensor view | `Tensor` | opaque "
                "`chelis_tensor *` handle and `CHELIS_VALUE_TENSOR` |",
                "tensor carrier mapping",
            ),
            (
                "| `Option<T>` where `T` has a target recursive-value representation "
                "| `Option` | opaque `chelis_option *` handle and "
                "`CHELIS_VALUE_OPTION` |",
                "Option carrier mapping",
            ),
            (
                "| `MappedFile` resource | `MappedFile` | opaque "
                "`chelis_mapped_file *` handle and `CHELIS_VALUE_MAPPED_FILE` |",
                "mapped-file carrier mapping",
            ),
            (
                "`CHELIS_VALUE_MAPPED_FILE` is the exact tagged representation "
                "when that handle\nis stored in `Option`, `List`, tuple, "
                "dictionary, or ADT",
                "recursive mapped-file representation",
            ),
            (
                "tensor storage is the sole private heap allocation with no public "
                "tag. Every\ndirectly carried public heap kind also has the table's "
                "exact tagged\nrepresentation for recursive aggregates",
                "public heap tag totality",
            ),
            (
                "Each identity has\nexactly one disposition: structurally nonheap, "
                "target-rejected with an owning\ncapability issue, direct heap "
                "carrier, tagged heap payload, or private heap\nallocation",
                "closed target-rejection disposition",
            ),
            (
                "A function\nstored in `Option`, `List`, tuple, dictionary, or ADT "
                "is a `FirstClassValue`,\nnot a contextual callback",
                "recursive function placement",
            ),
            (
                "`UnsupportedKind::HostAbi`, `Stage::Codegen(\"c\")`, and\n"
                "`Unimplemented { issue: #879 }` after the sealed ownership "
                "boundary certifies\nthe exact selected payload and before backend "
                "emission",
                "recursive function target rejection",
            ),
            (
                "It is a target capability result, not a language type error, "
                "scalar\nsubstitution, empty value, or permission to omit the type "
                "from the registry",
                "function rejection semantics",
            ),
            (
                "`chelis_option_scalar` / `chelis_option_value` split",
                "Option legacy-carrier deletion target",
            ),
            (
                "The exact ABI is [05-OP-31..33], [05-OP-44], and all four registries",
                "complete C ABI authority chain",
            ),
            (
                "A successful begin invalidates every previously returned read view; "
                "dereferencing\n  such a stale view violates the caller precondition",
                "write-begin read-view invalidation",
            ),
            (
                "This phase promotes exactly twenty-three oracle rows: the five [#543]\n"
                "aggregate-tensor rows (including the function-internal tensor-literal\n"
                "temporary), the eight [#544] size/nesting rows, the five direct/nested\n"
                "`Option` and mapped-file rows, and the five [#879] Metal rejection rows",
                "Phase 1 exact ownership row map",
            ),
            (
                "This phase promotes exactly six oracle rows: the [#1346] fold row, "
                "the two\n[#1352] mixed fresh-arm rows, the two [#1356] "
                "fresh-argument rows, and\n`recursive-depth-1-control`",
                "Phase 2 exact ownership row map",
            ),
            (
                "balanced tensor/string/List/tuple/dictionary/ADT/Option/mapped-file "
                "ownership,\n   including `Option[MappedFile]`, nested resource "
                "aggregates",
                "Option ownership fixtures",
            ),
            (
                "omit `ConcreteHostType::Option` or `CHELIS_VALUE_OPTION`",
                "Option omission mutation",
            ),
            (
                "omit `CHELIS_VALUE_MAPPED_FILE` or its `Option[MappedFile]` "
                "fixture",
                "mapped-file omission mutation",
            ),
            (
                "admit `Option[function]` or another recursive function container "
                "without the\n  exact [#879] target rejection",
                "function-container omission mutation",
            ),
            (
                "Only the planner constructs `ReusableOwnedStorage`",
                "private reuse proof",
            ),
            (
                "C and HIP consume\nthe same proof-bearing plan",
                "shared C and HIP reuse proof",
            ),
            (
                "typed `MetalNeverReuse` plan whose input cannot carry "
                "`ReusableOwnedStorage`",
                "Metal typed no-reuse plan",
            ),
            (
                "Metal emission with distinct storage for every produced node and "
                "no input",
                "Metal no-alias fixture",
            ),
            (
                "let the Metal plan accept `ReusableOwnedStorage`",
                "Metal no-reuse mutation",
            ),
            (
                "scripts/compiled_value_ownership_oracle.py --phase launch",
                "launch ownership oracle command",
            ),
            (
                "COMPILED VALUE OWNERSHIP LAUNCH SUBSET: PASS",
                "launch ownership oracle success line",
            ),
            (
                "The `complete --require-hip` invocation is the eventual [#1286] "
                "class-closure\noracle. It is deliberately stronger than the "
                "launch invocation",
                "launch and class-closure distinction",
            ),
            (
                "A support, syntax, diagnostic, or reachability observation outside\n"
                "   this class must have an explicit external owner before phase "
                "exit",
                "external observation ownership",
            ),
            (
                "The top-level tuple missing-`main` observation is [#545], not an "
                "ownership-oracle row",
                "top-level tuple external owner",
            ),
            (
                "Recursive-host operation support remains [#729]/[#730] capability "
                "work",
                "recursive support external owners",
            ),
            (
                "[#1172] owns the span-key cause that can over-broaden hints; Surf "
                "reachability is exposure evidence",
                "reachability external owner",
            ),
            (
                "Phase 1 must\n  explicitly supersede its numbered-spec citations, "
                "`runtime_representation.md`\n  target, guards, and public-layout "
                "promise in the same atomic change",
                "runtime representation supersession",
            ),
            (
                "[#909]/[#879]:** own shared first-class function representation "
                "and the\n  general C-host closure ABI",
                "function-value external owners",
            ),
            ("final manifest contains zero expected failures", "zero expected failures"),
            ("`Part of #1286`", "honest issue linkage"),
        ),
        violations,
    )
    require_all(
        implicit_linearity,
        (
            (
                "The verified `OwnershipProgram` makes `RiscOp::Copy` and "
                "`RiscOp::Drop` real\nownership operations",
                "current Drop implementation status",
            ),
            (
                "C and HIP emit the exact\ndescriptor release selected by the "
                "verified directive; Metal consumes the same\ndirective as a typed "
                "no-device-owner disposition",
                "successor Drop release",
            ),
        ),
        violations,
    )
    spec04_blocks = atom_blocks(spec04)
    require_atom(
        spec04_blocks,
        "04-LIN-3",
        (
            "exactly one logical owner",
            "exactly one terminal consuming use or `Drop`",
            "a borrow neither creates nor terminates an owner",
        ),
        violations,
    )
    require_atom(
        spec04_blocks,
        "04-LIN-4",
        (
            "owned function parameter is a consuming call edge",
            "every return path",
            "result owner may be the owner transferred through an owned parameter",
            "borrowed argument or still-live capture",
            "ordinary copy operation creates an independent owner",
            "pointer equality, a source name, or a selected return arm",
        ),
        violations,
    )
    require_atom(
        spec04_blocks,
        "04-LIN-5",
        (
            "exactly one owned incoming value from every predecessor path",
            "Fold and loop-carried owners are block parameters",
            "Alias provenance SHALL NOT be overwritten",
        ),
        violations,
    )
    require_atom(
        spec04_blocks,
        "04-LIN-6",
        (
            "in manifest order",
            "implicit terminal consuming use",
            "participates in the same copy insertion",
        ),
        violations,
    )
    require_atom(
        spec04_blocks,
        "04-LIN-7",
        (
            "externally supplied entry arguments are borrowed",
            "neither mutate nor release their storage",
            "Before an entry value crosses an internal owned-parameter edge",
            "compiler SHALL create an ordinary copy",
            "independent owner for the caller",
        ),
        violations,
    )
    require_atom(
        spec04_blocks,
        "04-LIN-8",
        (
            "move may transfer the value to one explicit successor owner",
            "consuming use with no successor owner",
            "reclaimable before a following tail call or loop back-edge",
            "proved unique",
            "recursion depth alone is not such a reason",
        ),
        violations,
    )
    require_atom(
        spec04_blocks,
        "04-NUM-2",
        (
            "IEEE-754 round-to-nearest, ties-to-even, at the dtype's own STORAGE "
            "width",
            "canonical quiet NaN: f16 `0x7e00`, bf16 `0x7fc0`, f32 "
            "`0x7fc00000`, or f64 `0x7ff8000000000000`",
            "Arithmetic preserves the NaN class, not an input payload or sign",
            "A pure bit-moving or selection operation preserves NaN payload bits "
            "only when its governing operation atom explicitly says it is "
            "bit-preserving",
        ),
        violations,
    )
    require_atom(
        spec04_blocks,
        "04-NUM-8",
        (
            "Every dtype declares a STORED REPRESENTATION and an ARITHMETIC WIDTH",
            "Equal storage widths do not make two representations interchangeable",
            "Every boundary and lane SHALL match the exact representation identity",
            "No implementation may infer arithmetic width from storage width or "
            "storage width from arithmetic width",
        ),
        violations,
    )
    require_atom(
        spec04_blocks,
        "04-NUM-11",
        (
            "A language binding or device descriptor SHALL preserve rank as i32 "
            "and each extent, stride, element count, and byte capacity as i64",
            "It SHALL carry the exact dtype tag and dynamic rank",
            "a fixed-rank carrier, a narrower metadata field, or an element pointer "
            "not coupled to the exact tag in the same validated descriptor is not a "
            "conforming substitute",
            "it SHALL NOT narrow, clamp, wrap, or fabricate metadata to make it fit",
        ),
        violations,
    )
    require_atom(
        spec04_blocks,
        "04-SHAPE-1",
        (
            "sound over the exact mathematical values of the complete typed extent "
            "expressions",
            "SHALL NOT wrap, saturate, truncate, or substitute an overflow sentinel "
            "that can make unequal mathematical counts equal",
            "Projection from the exact count into `i64`, `usize`, or a target "
            "allocation-size domain SHALL be checked",
            "a reuse decision is not exempt because no bytes have yet been touched",
        ),
        violations,
    )
    require_atom(
        spec04_blocks,
        "04-NUM-16",
        (
            "active signed-integer or float source",
            "signed-integer target",
            "truncates a finite float toward zero",
            "`-inf` clamps to the target minimum",
            "`NaN` traps `Domain`",
            "congruent to the exact source modulo",
            "A rounding cast is not a separate operation",
            "`round(source)` followed by checked `cast`",
            "[04-NUM-15] governs the selected failure",
        ),
        violations,
    )
    require_all(
        spec06,
        (
            (
                "over these) is not batched",
                "vmap runtime extent non-batching rule",
            ),
            (
                "### 8.6 `batch_varying_extent` (vmap)",
                "vmap batch-varying extent rejection",
            ),
            (
                "If `A = List[T]` and `dT` is defined, then `dA = List[dT]`; the "
                "cotangent\n  list has exactly the primal list's runtime length and "
                "positional order",
                "recursive List cotangent",
            ),
            (
                "If `A` is an ADT/record, then `dA` has the same executed constructor "
                "shape",
                "recursive ADT cotangent",
            ),
            (
                "never drops a tuple field,\nlist element, or ADT field merely because "
                "its cotangent is unit",
                "shape-preserving recursive cotangent",
            ),
            (
                "`match` differentiates the arm executed by the forward program",
                "match executed-arm adjoint",
            ),
            (
                "Scalar `if` likewise differentiates the executed branch and gives its "
                "boolean\ncondition zero cotangent",
                "scalar-if executed-branch adjoint",
            ),
            (
                "Recursive calls differentiate the finite recurrence actually executed "
                "by the\nforward program and reverse that recorded call trajectory",
                "recursive-call adjoint",
            ),
            (
                "A missing host-ABI\ncarrier is a backend capability gap, not a language "
                "restriction",
                "recursive AD target independence",
            ),
            (
                "Potentially effectful or trapping nodes are observable roots; purity alone "
                "does\nnot make a possible trap dead",
                "optimizer observable roots",
            ),
            (
                "none of these\npatterns is unconditional",
                "optimizer proof obligation",
            ),
            (
                "Fusion preserves every primitive's declared arithmetic width and "
                "stored-value\nfinalization boundary",
                "fusion finalization boundary",
            ),
            (
                "upstream = balanced_sum(\n"
                "            exact_zero(cotangent_type(type_of(n))),",
                "formal balanced cotangent accumulation",
            ),
            (
                "canonical forward node ordinal, then by input-slot index",
                "canonical consumer-edge order",
            ),
            (
                "independent of the work-list or topological-sort tie order",
                "topological-sort independence",
            ),
            (
                "stable_topological_order(N, tie_break=canonical_forward_ordinal)",
                "stable formal traversal",
            ),
            (
                "values_sorted_by_key(contributions[n])",
                "key-sorted formal accumulation",
            ),
            (
                "contributions[n_j][(canonical_forward_ordinal(n_i), input_slot)] =\n"
                "    contribution_from_n_i",
                "keyed backward-traversal contribution queue",
            ),
            (
                "return pack_wrt_gradients(grads)",
                "formal gradient-only result",
            ),
        ),
        violations,
    )
    require_all(
        spec04,
        (
            (
                "axis arguments are sorted by their positions in the original operand, "
                "from\nhighest position to lowest",
                "canonical variadic reduction order",
            ),
            (
                "composition owns the exact value, trap, NaN-selection, and adjoint "
                "behavior",
                "canonical composition authority",
            ),
            ("sum_result(p,a)", "sum result precision rule"),
            (
                "`sum_result(p, a) = p` exactly when `p` is `bf16` or `f16`;\n"
                "otherwise `sum_result(p, a) = a`",
                "total sum result precision rule",
            ),
            (
                "`i32` | `i32`, `i64` | accumulator dtype `a`",
                "explicit wider integer accumulator result",
            ),
            (
                "An accumulator has the same numeric kind as its operands",
                "same-kind accumulator rule",
            ),
            (
                "Every literal or computed\naxis first adds the input rank exactly once when negative",
                "shape negative-axis normalization",
            ),
            (
                "A statically known normalized value outside `0..rank` is a type\n"
                "error (`DimensionMismatch`)",
                "shape post-normalization rejection",
            ),
            (
                "access whose shape depends on the guarded extent",
                "runtime extent guard placement",
            ),
            (
                "places guards by this rule",
                "runtime extent guard placement in every execution mode",
            ),
            (
                "and the dtype of the quantity that guard\n> finalizes",
                "precondition guard finalized-quantity dtype",
            ),
            (
                "`numeric trap: domain in <op> at i64`",
                "runtime extent guard trap line",
            ),
            (
                "Each operation has exactly one result shape. `expand` sets "
                "the extent at\n`axis` and leaves the rank unchanged; `insert` adds an axis of extent `size`\nat `axis` and produces rank `rank(x) + 1`. No result is deferred, no consumer\nselects between shapes, and no context supplies a default.",
                "expand and insert each have one result shape",
            ),
            (
                "`insert` admits `axis` in `0..=rank(x)`, so\n`axis == rank(x)` appends a trailing axis. An axis outside its operation's\nrange is a type error.",
                "insert axis range",
            ),
            (
                "**Named-axis insert (`R+1`).** The inverse arithmetic "
                "direction: `insert`\nadds a *named* axis",
                "named-axis form belongs to insert",
            ),
            (
                "| Ordered comparison (`cmplt`, `lt`, `gt`, `gte`, `lte`) | any "
                "active numeric dtype (both operands same dtype) → bool |",
                "closed ordered-comparison precision row",
            ),
            (
                "| Equality (`eq`, `neq`) | any active numeric dtype or bool (both "
                "operands same dtype), plus the recursively comparable host-value "
                "domain in [05-OP-36] → bool |",
                "closed equality precision row",
            ),
            (
                "Every tensor comparison preserves the operand surface: two "
                "same-dtype tensors\nwith identical dimensions return a bool tensor "
                "with those dimensions",
                "comparison surface preservation",
            ),
            (
                "Numeric\nor bool scalar equality returns a bool scalar; ordered "
                "scalar comparison is\nnumeric only",
                "scalar equality and ordered-comparison split",
            ),
            (
                "Mixed tensor/scalar surfaces or tensor dimensions are type errors.\n"
                "[05-OP-36] owns the exact values and the complete equality domain",
                "comparison surface rejection and authority",
            ),
            (
                "`cumsum(x, axis=k)`",
                "cumsum result precision rule",
            ),
            (
                "`trace(x, axis1, axis2)`",
                "trace result precision rule",
            ),
            (
                "`einsum(equation, left, right, accumulator=a)`",
                "einsum result precision rule",
            ),
            ("A typed operation-precondition guard", "typed reduction guard"),
            (
                "`count` is the bool-tensor counting operation",
                "bool count operation",
            ),
        ),
        violations,
    )

    atom_requirements: dict[str, tuple[str, ...]] = {
        "05-OP-1": (
            "nearest to the EXACT binary value of `x`",
            "finalized ONCE to the operand's own storage width",
            "| `f64` | exact decimal rounding of the exact binary value, one final "
            "rounding to f64 | `f64` |",
            "| `f32` | exact decimal rounding of the exact binary value, one final "
            "rounding to f32 | `f32` |",
            "| `f16` | exact decimal rounding of the exact binary value, one final "
            "rounding to f16 | `f16` |",
            "| `bf16` | exact decimal rounding of the exact binary value, one final "
            "rounding to bf16 | `bf16` |",
            "| integer, bool, tensor | type error | n/a |",
            "piecewise constant",
            "structurally rejected with `AdRejectionReason::PiecewiseConstant`",
            "rather than receiving a silent zero cotangent",
            "has no accumulator",
        ),
        "05-OP-2": (
            "A JSON number token containing `.`, `e`, or `E`",
            "ingest as `JsonFloat` carrying the correctly-rounded f64 of the token",
            "any other number token",
            "ingest as `JsonInt` carrying its exact i64 value",
            "An integer-form token outside i64 range SHALL ingest as",
            "`JsonBigInt` carrying the token's exact decimal spelling",
            "never\n> selects a lossy float image for an integer-form token",
            "CSV cells are TEXT at parse time",
            "integer accessors accept only its integer subset",
            "An empty or non-conforming cell is a loud error",
        ),
        "05-OP-3": (
            "`io/json::json_int` returns the stored `JsonInt` i64 exactly",
            "`io/json::json_float` returns a stored `JsonFloat` f64 exactly",
            "It never truncates or rounds a float into an integer",
            "`csv_int` | `(List[Dict[string,string]], i64, string) -> i64`",
            "`csv_ints` | `(List[Dict[string,string]], string) -> List[i64]`",
            "`csv_f64` | `(List[Dict[string,string]], i64, string) -> f64`",
            "`csv_f64s` | `(List[Dict[string,string]], string) -> List[f64]`",
            "`csv_nrows` | `(List[Dict[string,string]]) -> i64`",
            "no JSON variant or default cell is fabricated",
            "structurally rejected inside `grad`",
            "They have no accumulator",
        ),
        "05-OP-4": (
            "`JsonFloat(value)` accepts exactly f64",
            "`JsonInt(value)` accepts exactly i64",
            "every other operand width is a type error",
            "No construction path widens or narrows a numeric value",
            "feeds the byte-exact serialization channel of [05-OP-5]",
        ),
        "05-OP-5": (
            "emits a stored `JsonInt` i64 as its exact decimal digits",
            "a stored f64 through the [05-OBS-1]",
            "every finite emission parses back to the identical f64",
            "A non-finite `JsonFloat` is a loud serialization error",
            "Equal documents serialize to identical bytes",
            "`to_csv` accepts only the text-table type `List[Dict[string,string]]`",
            "without inferring, preserving, or serializing a numeric cell type",
            "Numeric source values enter CSV only through explicit `to_string`",
        ),
        "05-OP-6": (
            "explicit truncating narrowing cast from a float source dtype to an "
            "integer target dtype",
            "**finite** source value it yields the integer part truncated toward zero",
            "outside the target range it traps `overflow`",
            "A **non-finite** source (`NaN`, `±inf`) traps `Domain`",
            "not float→integer, `cast_trunc` is a type error",
            "never widens, never rounds, and never applies to `bool`",
            "Semantics are identical on scalar and tensor surfaces",
            "a gradient goal through it is a clean error, never a silent zero",
            "has no accumulator",
        ),
        "05-OP-7": (
            "returns the stored extent of `x` along `axis` as an exact `i64`",
            "a negative value first normalizes by adding the rank exactly once",
            "an axis still outside `0..rank` is a loud error",
            "read has a zero-cotangent adjoint",
            "contributes exact zero to the tensor input and to the discrete axis",
            "does not block differentiation of a surrounding graph",
        ),
        "05-OP-8": (
            "admits every active float template dtype `p` in spec/04 §1.1",
            "requires `low` and `high` to have that same dtype `p`",
            "returns `tensor[D, p]` with the template's dimensions",
            "Both bounds must be finite and `low <= high`",
            "At the selected arithmetic width, `high - low` must also be finite",
            "consumes the key `k`",
            "complete before any element is drawn",
            "failure traps `Domain` as `uniform_like` before any element is produced",
            "Equal bounds are valid and produce that stored value",
            "For `p = f64`, the element is the one f64 fused multiply-add",
            "For `p = f32`, it is the one f32 fused multiply-add",
            "For `p = f16` or `bf16`, the stored bounds widen exactly to f32",
            "their difference and the fused multiply-add execute once in f32",
            "the result narrows exactly once to `p`",
            "There is no f32 public-bound signature, default bound, or f64 "
            "intermediate",
            "does not observe the template's element values, and the key carries no "
            "cotangent",
            "pathwise adjoint contributes zero to the template",
            "contributes `g_i * (1-u_i)` to `low` and `g_i * u_i` to `high`",
            "canonical adjacent-pair balanced tree",
            "has no accumulator parameter",
        ),
        "05-OP-9": (
            "admits every active tensor element dtype `T` in spec/04 §1.1, "
            "including `bool`",
            "Every source and padding element is moved at its declared dtype `T`",
            "no arithmetic, widening, narrowing, or other rounding",
            "differentiable when `T` is a float dtype",
            "source cotangent preserves the outer and inner runtime List shapes "
            "exactly",
            "source element `(r, c)` receives `g[r, c]`",
            "sum of `g[r, c]` over padded result cells in increasing row-major "
            "`(r, c)` order",
            "canonical adjacent-pair balanced tree",
            "exact positive-zero base leaf",
            "When there are no padded cells that cotangent is positive zero",
            "For integer or bool `T`, the operation is forward-only",
            "has no public accumulator parameter",
        ),
        "05-OP-10": (
            "has the same dtype, element-movement, float-domain differentiation, and "
            "no-public-accumulator rules as [05-OP-9]",
            "`width` SHALL be non-negative",
            "source elements at index `width` or beyond do not appear",
            "a source element `(r, c)` receives `g[r, c]` when `c < width` and "
            "exact positive zero otherwise",
            "truncated source cells remain present in the nested List cotangent",
            "`pad` cotangent uses [05-OP-9]'s exact traversal, arithmetic, tree, "
            "and positive-zero rule",
            "`width` is non-differentiable",
        ),
        "05-OP-11": (
            "`f16`, `bf16`, `f32`, or `f64`",
            "composition `div(sum(x, axis), divisor)`",
            "converted once to the sum result dtype",
            "zero-length axis is a type error when statically known",
            "execution-time extent is zero",
            "traps `Domain` as operation\n> `mean` at the result dtype",
            "`insert(g / divisor, axis, original_extent)`",
            "no accumulator parameter of its own",
        ),
        "05-OP-12": (
            "every active signed\n> integer and float tensor dtype",
            "compared without conversion at their stored dtype",
            "first NaN in increasing axis-index order",
            "first stored representation among equal\n> values",
            "execution-time extent is zero",
            "equal positive or negative infinities",
            "tie count `k` is\n> counted exactly as `i64`",
            "`div(g, k)`",
            "full cotangent flows to the\n> first NaN",
            "Integer operands are forward-only and `grad` rejects them",
        ),
        "05-OP-13": (
            "replacing maximum by minimum",
            "first NaN in\n> increasing axis-index order",
            "equal positive or negative\n> infinities",
            "upstream cotangent divided by the number of equal\n> minima",
        ),
        "05-OP-14": (
            "canonical balanced tree",
            "pairs adjacent values from left to right",
            "carries an\n> odd final value unchanged",
            "finalized once to the operand storage dtype before it enters the next level",
            "overflow is checked at every multiplication",
            "zero-length axis returns the multiplicative identity",
            "reverse-mode\n> derivative of that exact multiplication tree",
            "gradients at\n> zero operands are defined",
            "`prod_reduce` never treats `bool` as integer",
        ),
        "05-OP-15": (
            "every active\n> signed integer and float tensor dtype",
            "returns `i64` indices",
            "lowest\n> axis index containing NaN",
            "Comparisons never convert through another dtype",
            "execution-time extent is zero",
            "result dtype `i64`",
            "non-differentiable: `grad` rejects it",
        ),
        "05-OP-16": (
            "contract of [05-OP-15]",
            "lowest NaN index when present",
            "stored value is minimal",
        ),
        "05-OP-17": (
            "two signed-integer\n> scalar operands or two signed-integer tensor operands",
            "congruent to the exact mathematical sum modulo `2^w`",
            "never traps for\n> overflow",
            "`grad` rejects it",
        ),
        "05-OP-18": (
            "contract of [05-OP-17]",
            "exact mathematical difference modulo `2^w`",
        ),
        "05-OP-19": (
            "contract of [05-OP-17]",
            "exact mathematical product modulo `2^w`",
        ),
        "05-OP-20": (
            "`f16`, `bf16`, `f32`, or\n> `f64` scalar or tensor",
            "returns `bool` on the same surface",
            "finalized stored value\n> without conversion",
            "true exactly when that value is NaN",
            "contributes zero cotangent",
            "Integer, `bool`, `string`, and reserved dtype spellings are type errors",
        ),
        "05-OP-21": (
            "contract of\n> [05-OP-20]",
            "true exactly for finite stored values",
            "false for NaN and both infinities",
        ),
        "05-OP-22": (
            "contract of\n> [05-OP-20]",
            "true exactly for positive or negative infinity",
            "false for NaN and every finite value",
        ),
        "05-OP-23": (
            "active\n> signed-integer or float source dtype",
            "signed-integer target dtype",
            "reads the source exactly at its\n> stored dtype",
            "finite float is truncated toward zero",
            "Negative infinity returns the target minimum",
            "NaN traps `Domain`",
            "never traps `Overflow`, wraps, or\n> converts through another numeric dtype",
            "`grad` rejects it",
        ),
        "05-OP-24": (
            "signed-integer source and signed-integer target",
            "congruent to the exact stored\n> source modulo `2^target_width`",
            "never traps for overflow, saturates, or\n> converts through a float dtype",
            "`grad` rejects it",
            "There is no `cast_round` operation",
            "`cast(round(source), target)`",
        ),
        "05-OP-25": (
            "`to_string(value) -> result` borrows exactly one value",
            "without consuming it and returns `string`",
            "It admits exactly an active numeric, `bool`, or `string` scalar",
            "a tensor whose element dtype is one of the nine active data element dtypes",
            "a `List` whose reachable elements are recursively admitted by this rule",
            "Unit, tuples, `Dict`, `Option`, ADTs, functions, resource handles, and "
            "deferred values are type errors",
            "returns `value` byte-for-byte unchanged",
            "dimensions `[d0, ..., d_(r-1)]` and `N` elements",
            "all `N` elements when `N <= 32`, otherwise the first 32",
            "A List boundary never truncates or elides elements",
            "Each nested List element uses this same rule",
            "String elements are inserted verbatim, without quoting or escaping",
            "non-injective display form, not a serialization",
            "Every lane produces byte-identical text",
            "operation is pure",
            "performs no arithmetic or dtype conversion",
            "non-differentiable (`grad` rejects it)",
            "no accumulator",
        ),
        "05-OP-26": (
            "exactly two `bool` scalars or two `bool` tensors",
            "identical dimensions",
            "evaluate `left` and then `right`",
            "not short-circuiting",
            "true exactly when both operands are true",
            "performs no arithmetic or dtype conversion",
            "has no accumulator",
            "`grad` rejects it",
        ),
        "05-OP-27": (
            "contract of [05-OP-26]",
            "true exactly when either operand is true",
            "applied element-wise for tensors",
        ),
        "05-OP-28": (
            "exactly one `bool` scalar or `bool` tensor",
            "true exactly when `value` is false",
            "Any non-`bool` operand is a type error",
            "performs no arithmetic or dtype conversion",
            "has no accumulator",
            "`grad` rejects it",
        ),
        "05-OP-29": (
            "admits exactly a `bool` tensor operand",
            "returns an `i64` tensor",
            "one or more unique named axes",
            "Missing axes, mixed positional/named axes, duplicate normalized "
            "positions or names",
            "visited in original row-major order",
            "checked `i64` addition",
            "traps `Overflow` as operation `count`",
            "result is `0i64`",
            "dedicated reduction and is not a `cast` plus `sum` lowering",
            "not a composition of nested `count` calls",
            "Numeric, scalar `bool`, `string`, reserved dtype spellings",
            "no accumulator",
            "structurally rejected with `AdRejectionReason::IntegerReductionOutput`",
            "never receives a silent zero cotangent",
        ),
        "05-OP-30": (
            "active signed integer or active float",
            "`bool`, `string`, reserved dtype spellings, scalar, and all other operands are",
            "same numeric kind as `p`",
            "§5.7.1's default",
            "canonical balanced tree",
            "integer overflow is checked at every addition",
            "Empty slices return exact zero",
            "`sum_result(p, accumulator)`",
            "adjoint expands the upstream cotangent",
            "Signed-integer forms are forward-only",
            "stride-4 cascade",
            "bool-to-integer promotion",
        ),
        "05-OP-31": (
            "exactly the ten final public C callables",
            "CHELIS_VALUE_ADT = 7, CHELIS_VALUE_OPTION = 8, "
            "CHELIS_VALUE_MAPPED_FILE = 9 };",
            "typedef struct { const void *data; int64_t count; chelis_dtype "
            "dtype; uint8_t reserved[7]; } chelis_read_view;",
            "typedef struct { void *data; int64_t count; chelis_dtype dtype; "
            "uint8_t reserved[7]; } chelis_write_view;",
            "there is no by-value option carrier",
            "exactly one of the ten constants above",
            "that handle is exactly one [05-OP-44] owner",
            "`chelis_tensor` is [05-OP-44]'s opaque descriptor handle and has "
            "no public field",
            "typedef struct { chelis_value key; chelis_value value; } "
            "chelis_dict_entry;",
            "rank in `0..=INT32_MAX`",
            "a rank-zero descriptor has no extents",
            "exactly `rank` nonnegative i64 extents",
            "with the rank-zero empty product equal to one",
            "There is no rank-eight limit",
            "byte size is the checked product `count * chelis_dtype_size(dtype)`",
            "A view with zero `count` has null `data`",
            "A read view is valid only while an owner of its descriptor is live "
            "and until that descriptor is passed to "
            "`chelis_tensor_begin_write`, whichever comes first",
            "A successful begin invalidates every read view previously returned "
            "for that descriptor",
            "dereferencing such a stale view violates the caller precondition",
            "A write view is valid only while its exclusive guard is live",
            "nothing is retained, released, or freed through a view pointer",
            "Foreign storage enters only through [05-OP-44]'s entry borrow",
            "through [05-OP-44]'s exclusive write guard",
            "F32=0`, `F64=1`, `I32=2`, `Bool=3`, `I64=4",
            "unused high bits are zero",
            "bool payload is exactly `0` or `1`",
            "NaN payload and signed-zero bits",
            "Signed-decimal integer text is exactly an optional `+` or `-` "
            "followed by one or more ASCII digits",
            "a bare sign, internal whitespace, or non-ASCII digit is malformed",
            "A finite float token is an optional `+` or `-`",
            "optional exponent `[eE][+-]?[0-9]+`",
            "a bare sign, bare point, internal whitespace, non-ASCII digit, hex "
            "form, or suffix is malformed",
            "Its exact decimal value rounds once at the requested float width",
            "finite overflow returns `None` and underflow rounds normally, "
            "including to signed zero",
            "The only accepted NaN spelling is exactly `NaN`",
            "f16 `0x7e00`, bf16 `0x7fc0`, f32 `0x7fc00000`, and f64 "
            "`0x7ff8000000000000`",
            "does not convert through `double`",
            "rank-zero tensor with exactly one element",
            "`None` only when the key is absent",
            "no alias, wrapper, or deprecated spelling",
            "has no accumulator and is outside AD",
        ),
        "05-OP-32": (
            "exactly the container, extent, index, byte-read, and recursive-observation",
            "chelis_string chelis_string_slice(chelis_string value, int64_t start, int64_t len)",
            "chelis_list *chelis_list_from_values(const chelis_value *items, int64_t len)",
            "chelis_dict *chelis_dict_insert(const chelis_dict *dict, chelis_value key, chelis_value value)",
            "chelis_list *chelis_mmap_read(const chelis_mapped_file *mapped, int64_t offset, int64_t len)",
            "All lengths, indices, offsets, sizes, and returned counts are exact `i64`",
            "Negative lengths, indices, offsets, and counts trap `Domain`",
            "result-length and allocation arithmetic traps `Overflow`",
            "half-open increasing sequence",
            "Unicode scalar values",
            "A slice whose nonnegative start is at or beyond the scalar length "
            "is empty",
            "Length-aware UTF-8 construction copies exactly the declared bytes",
            "permits a null source pointer only for zero length",
            "Character-code conversion follows [05-OP-58] exactly",
            "Length-aware string observation writes every stored UTF-8 byte",
            "without C-string termination semantics",
            "Dictionary keys are exactly `string`, `bool`, or a scalar of any active "
            "signed-integer dtype",
            "Equality includes the key kind and integer dtype",
            "Float keys are rejected",
            "later duplicate replaces the value",
            "`dict_get` returns one owned [05-OP-44] option node that is `None` "
            "only for absence",
            "An option node renders as `None` when it owns no child and otherwise "
            "as `Some(` followed by `R` of its child and `)`",
            "A mapped file renders as `<mapped-file:` followed by its exact i64 "
            "byte length in decimal digits and then `>`",
            "Recursive dictionary observation is canonical rather than "
            "insertion-ordered",
            "Recursive observation uses one byte grammar `R(value)` over every "
            "validated `chelis_value` variant",
            "A tuple renders as `()` when it has no fields, `(R(v),)` when it has one",
            "A dictionary renders entries in the canonical key order above as",
            "`R(key): R(value)` pairs separated by `, `",
            "An ADT renders its exact stored constructor-name bytes followed by `(`",
            "a zero-field constructor therefore renders as `Ctor()`",
            "unit and an empty tuple both render `()`",
            "Each of `chelis_print_list`, `chelis_print_tuple`, "
            "`chelis_print_dict`, and `chelis_print_adt` writes exactly `R` of its "
            "argument",
            "followed by one byte `\\n` to standard output",
            "adds no label, prefix, extra space, truncation beyond the nested "
            "tensor rule, or additional newline",
            "A short or failed write traps `IO`",
            "Successful return means every required byte was written",
            "[05-OBS-1..5] at each stored scalar's own dtype",
            "outside AD and have no accumulator",
        ),
        "05-OP-33": (
            "returns exact i64 zero for a rank-zero input, or one when the input shape",
            "is identical to the domain shape",
            "It validates every extent and the exact zero-aware element product before",
            "An iteration domain requires neither storage byte counts nor contiguous suffix strides",
            "A caller validates the original input before repurposing its storage",
            "excluding spare storage capacity",
            "takes rank and every target extent as exact tagged i64 scalars",
            "changes no metadata, ownership, or payload",
            "preserves every stored element bit",
            "exactly the public C callable identities enumerated in",
            "unboxed axes and rank are `int32_t`",
            "host tensor arguments and results are [05-OP-44]'s opaque "
            "`chelis_tensor` handles, and every host tensor result is a new owner",
            "alignment, live-owner state, and write-guard state, before reading "
            "data",
            "Foreign host tensor storage enters only through [05-OP-44]'s entry "
            "borrow; host tensor callables in this family do not construct a "
            "non-owning host tensor view, adopt caller bytes, or free storage",
            "extents, sizes, offsets, counts, and element counts are `int64_t`",
            "rank is nonnegative",
            "before allocation or element access",
            "Every signed axis accepted by this C family first applies §2.3's "
            "one-step negative normalization",
            "an axis still out of range then traps `Domain`",
            "scalar leaves all have exactly the requested dtype",
            "`chelis_tensor_elements` boxes every element as its exact scalar in "
            "row-major order for every rank",
            "Rank zero therefore returns a one-element list",
            "any zero extent returns an empty list",
            "There is no rank-specialized or recursively nested list-egress alias",
            "Each destination's canonical balanced accumulation tree begins with "
            "an exact positive-zero base leaf",
            "not an omitted initializer",
            "`chelis_tensor_scatter_replace`",
            "`chelis_tensor_scatter_add`",
            "`gather` admits an index tensor of any active signed-integer dtype",
            "Scatter indices have any active signed-integer dtype",
            "interpreted at their exact stored mathematical values",
            "are zero-based",
            "must lie in the selected base-axis extent",
            "Any negative or out-of-range index traps `Domain` before any write",
            "no i32/i64-only dispatch exception",
            "Replace admits every active dtype, including bool",
            "admits exactly active signed-integer and float dtypes",
            "NaNs follow all non-NaNs",
            "`sum_result(p, default(p))`",
            "[05-OP-30]'s canonical balanced tree",
            "`diagonal` admits every active dtype including bool",
            "rejects a NaN bound or `lower > upper`",
            "[a-z]*,[a-z]*->[a-z]*",
            "same numeric kind as `p`",
            "reverse derivative of this exact multiply-and-balanced-add graph",
            "This atom's selection rule also governs exactly the language builtin",
            "`where(condition, then, else)` with signature",
            "`(&tensor[D,bool], &tensor[D,p], &tensor[D,p]) -> tensor[D,p]`",
            "selection copies the chosen stored bits without numeric conversion",
            "the condition has no cotangent",
            "No operation in this family converts a stored element through `double`",
            "supplies a compatibility alias",
        ),
        "05-OP-34": (
            "exported stdlib ADT identities enumerated in the normative registry",
            "`io/json::Json`",
            "`decimal::Decimal`",
            "`time::Date`",
            "`time::Duration`",
            "`tokenizer::Tokenizer`",
            "accepts every representable declared field tuple",
            "validation and normalization belong to named stdlib functions",
            "A public signature is numeric when any reachable field of an admitted ADT",
            "expands\n> nominal ADT definitions recursively to a fixed point",
            "scanning only primitives spelled directly in the signature\n> is nonconforming",
            "There is no second prelude JSON identity or constructor registry",
            "ordinary constructor and the executed matching arm preserve the recursive "
            "cotangent shape",
            "differentiable float fields receive their corresponding field cotangents",
            "non-differentiable fields carry `unit`",
            "`JsonFloat(x)` followed by an executed `JsonFloat(y)` match routes the "
            "cotangent of `y` to `x`",
            "integer-only ADTs naturally have only `unit` field cotangents",
            "constructors have no accumulator",
        ),
        "05-OP-35": (
            "exactly the eighty-four final exported stdlib numeric definitions",
            "`process::run` | `(string,List[string])->(i64,string,string)!{IO}`",
            "`contracts::normal_cdf` | `(p_float)->p_float`",
            "`init/random::normal_like` | "
            "`(key,&tensor[..r,p_float],p_float,p_float)->tensor[..r,p_float]`",
            "`tensor/construct::linspace` | "
            "`(p_float,p_float,i64)->tensor[n,p_float]`",
            "`tensor/construct::arange` | "
            "`(p_int,p_int)->tensor[n,p_int]`",
            "`sort::sort` | `(&tensor[..r,p_numeric],i32)->"
            "(tensor[..r,p_numeric],tensor[..r,i64])`",
            "`scalar::abs` | `(p_numeric)->p_numeric`",
            "`scalar::max` | `(p_numeric,p_numeric)->p_numeric`",
            "`scalar::min` | `(p_numeric,p_numeric)->p_numeric`",
            "`test::assert_close` | "
            "`(p_float,p_float,p_float,string)->unit!{Test}`",
            "`test::assert_close_tensor` | "
            "`(&tensor[..r,p_float],&tensor[..r,p_float],p_float,string)->unit!{Test}`",
            "`test::assert_eq` | `(Q,Q,string)->unit!{Test}`",
            "`test::assert_eq_tensor` | "
            "`(&tensor[..r,p],&tensor[..r,p],string)->unit!{Test}`",
            "`tensor/construct::stack` | "
            "`(List[tensor[..pre,..post,p]],i32)->tensor[..pre,rows,..post,p]`",
            "Every primitive-width intermediate in a graph whose contract names a dtype",
            "Decimal rational and calendar ordinal computations explicitly named as "
            "mathematical below use an exact internal domain",
            "integer primitive arithmetic is checked",
            "JSON access follows [05-OP-2..5]",
            "Numeric tokens follow [05-OP-2]",
            "In this family `p` ranges over all active tensor element dtypes",
            "`p_numeric` over all active numeric dtypes",
            "`p_int` over all active signed integers",
            "`p_float` over all four active floats",
            "`Q` over one static type in [05-OP-36]'s scalar or recursive equality "
            "domain",
            "direct tensor arguments use `assert_eq_tensor`",
            "Every repeated variable denotes one common static type",
            "`standard_contract_tolerance` is deliberately the fixed f32 tolerance "
            "policy of the named standard-contract property corpus",
            "not an arithmetic operand, default dtype, or restriction on "
            "`normal_cdf`",
            "`normal_cdf(+inf)` is exact `1p`",
            "`normal_cdf(-inf)` is exact `0p`",
            "a NaN input returns [04-NUM-2]'s canonical NaN at `p_float`",
            "infinities have zero cotangent and NaN propagates the canonical NaN "
            "cotangent",
            "no non-finite input traps `Domain`",
            "A leading U+FEFF byte-order mark is not RFC 8259 whitespace and is "
            "rejected",
            "Every finitely nested valid document is in the language",
            "there is no fixed semantic nesting depth such as 512",
            "`scalar::abs` follows the unary abs rule at its active signed-integer "
            "or float dtype",
            "checked overflow at the signed minimum",
            "Scalar min/max return the first NaN with its exact stored payload and "
            "sign bits",
            "preserve the first operand on every equality, including signed-zero "
            "equality",
            "`arange(start,stop)` admits one active signed-integer dtype `p_int` "
            "for both endpoints",
            "returns the increasing half-open same-dtype sequence",
            "Its length and every step are checked in exact mathematical integers",
            "an unrepresentable length or element traps `Overflow`",
            "requires finite endpoints and i64 `count >= 1`",
            "Squeeze removes the selected singleton dimension",
            "Unsqueeze inserts a singleton dimension and stack inserts the "
            "input-list length at the selected position",
            "all three are rank-polymorphic, bit-preserving reshape/concat "
            "operations",
            "Squeeze normalizes a negative axis by adding the input rank once",
            "requires `0 <= axis < rank`; its selected extent must be one",
            "Unsqueeze and stack normalize a negative insertion axis by adding "
            "the result rank once",
            "require `0 <= axis <= input_rank`",
            "The float `squeeze` and `unsqueeze` adjoints are the reverse reshape "
            "graph",
            "The float `stack` adjoint slices the output cotangent along the "
            "inserted axis",
            "returns a `List` of tensor cotangents with exactly the input list's "
            "length, shapes, and dtype",
            "`linspace` with count one, start receives the sole output cotangent "
            "and stop receives exact zero",
            "enumerated by increasing output index",
            "separate canonical adjacent-pair balanced trees",
            "Kaiming requires finite `fan_in > 0`",
            "rejects a zero divisor or negative result scale",
            "interpreted in exact arithmetic and normalized before either "
            "representation check",
            "removable trailing zeros do not cause `Overflow`",
            "proleptic Gregorian calendar",
            "lowest merge rank and then the leftmost pair",
            "NaN is unequal to every value, including itself",
            "Test tolerances have the same active float dtype as the values",
            "`assert_close_tensor` admits exactly one common active float dtype `p`",
            "comparison executes at `p`'s [04-NUM-8] arithmetic width",
            "stored same-dtype tolerance converted exactly to that arithmetic width",
            "never converts either tensor through f64",
            "Scalar `assert_close` applies the same own-width rule to any active "
            "float dtype",
            "`assert_eq` uses [05-OP-36] equality for one common scalar or "
            "recursively comparable `Q`",
            "every container/ADT field follows the exact recursive rule",
            "Direct tensor arguments use `assert_eq_tensor`",
            "requires equal shapes and one common active element dtype",
            "signed-integer and bool elements use exact equality",
            "`assert_shape` requires its expected list to contain only nonnegative "
            "i64 extents",
            "compares its length and every entry to the tensor's complete shape "
            "in axis order",
            "Each finite element pair computes `abs(actual - expected)` at that "
            "width and is close exactly when that difference is less than or equal "
            "to the converted tolerance",
            "without invoking a shell",
            "every random stdlib callable has the pathwise adjoint of its exact "
            "graph above",
            "the key, source units, and mask comparisons contribute zero cotangent",
            "splits `k` by [05-OP-70] into `(k1, k2)`",
            "rounded result equals the stored upper endpoint",
            "computed denominator must be finite and strictly positive",
            "`days_between(lhs,rhs) = ordinal(rhs) - ordinal(lhs)`",
            "final normalized `days` field has no i64 representation",
            "A negative year uses `-` followed by exactly "
            "`max(4, digits(|year|))` decimal digits",
            "`|year|` is the exact mathematical magnitude rather than an i64 `abs`",
            "no token pair occurs at more than one merge rank",
            "repeatedly selects the lowest merge rank and then the leftmost pair",
            "`encode` maps each final token through `vocab`",
            "`decode` maps each ID through the inverse vocabulary",
            "No callable derives authority from its implementation body or age",
        ),
        "05-OP-36": (
            "exactly the seven language identities `cmplt`, `lt`, `eq`, `neq`, "
            "`gt`, `gte`, and `lte`",
            "The five ordered identities `cmplt`, `lt`, `gt`, `gte`, and `lte`",
            "two active-numeric scalars of one dtype",
            "two tensors of one active numeric dtype and identical dimensions",
            "`eq` and `neq` additionally admit bool scalars and same-shaped bool "
            "tensors, string scalars, unit",
            "two `List`, tuple, `Dict`, `Option`, or ADT values of one static type",
            "Functions and resource handles are not "
            "equality-comparable",
            "Ordered comparison of bool, string, or a structured value is a type "
            "error",
            "Scalar and recursive equality return one bool scalar",
            "tensor comparisons are element-wise and return a bool tensor",
            "Mixed surfaces, numeric dtypes, static structured types, or tensor "
            "dimensions are type errors",
            "Signed integers use exact mathematical order at their stored width",
            "Bool and string equality compare their exact stored values without "
            "Unicode normalization",
            "Dictionaries compare key/value sets independent of insertion order",
            "any NaN makes `cmplt`, `lt`, `eq`, `gt`, `gte`, and `lte` false "
            "and makes `neq` true",
            "signed zeros equal",
            "`neq` is the logical complement of `eq` for every non-float admitted "
            "value",
            "may use [05-OP-20] plus [05-OP-26..28] without sending bool through "
            "arithmetic IR",
            "These seven operations remain distinct typed comparison identities "
            "through AD and other semantic transforms",
            "logical expansion may occur only after the zero-cotangent adjoint is "
            "registered for the exact identity",
            "[05-OP-26..28]'s structural `grad` rejection does not replace this "
            "family's adjoint",
            "No identity has an alias, grandfathered path, deprecated spelling, or "
            "compatibility wrapper",
            "performs no numeric conversion, has no accumulator",
            "contributes zero cotangent to every differentiable leaf",
        ),
        "05-OP-37": (
            "admits every active float dtype `p`",
            "requires `input: &tensor[D,p]` and a scalar `rate: p`",
            "rate must be finite and satisfy `0 <= rate < 1`",
            "consumes the key `k`",
            "validation completes before any element is drawn",
            "failure traps `Domain` as `dropout` before any element is produced",
            "accepted call consumes its key",
            "including for an empty tensor or `rate = 0`",
            "saved forward mask drops the element exactly when that value is less "
            "than the rate",
            "A dropped element is positive zero at `p`",
            "A kept element computes the exact graph `denom = sub(1p, rate)` then "
            "`div(input[i], denom)`",
            "For f16 and bf16, `sub` exact-widens its stored operands to f32 and "
            "narrows its result to `p`",
            "`div` then exact-widens the stored `input[i]` and `denom` to f32 and "
            "narrows its result to `p`",
            "no f32 public-rate signature, f64 funnel, unscaled-dropout alias, or "
            "special `rate >= 1` default exists",
            "pathwise adjoint reuses the exact saved mask",
            "the language does not differentiate that selection",
            "through a data-flow path on which every operand slot has an adjoint "
            "contract",
            "a path through a zero-cotangent slot or a [05-OP-42] `stop_gradient` "
            "does not count",
            "`AdRejectionReason::RandomSelectionParameter`; otherwise the rate "
            "receives the exact zero cotangent",
            "mask comparison itself has zero cotangent",
            "has no accumulator parameter",
        ),
        "05-OP-38": (
            "governs exactly these five numeric-capacity identities and signatures",
            "`tensor_scan` | `(T,((T,i64)->T!E),i64)->tensor[n,..state_shape(T),element(T)]!E`",
            "`process_run` | `(string,List[string])->(i64,string,string)!{IO}`",
            "`test_assert_eq` | `(Q,Q,string)->unit!{Test}`",
            "`test_assert_close_tensor` | `(&tensor[..r,p_float],"
            "&tensor[..r,p_float],p_float,string)->unit!{Test}`",
            "`test_assert_eq_tensor` | `(&tensor[..r,p],&tensor[..r,p],string)"
            "->unit!{Test}`",
            "`T` is a scalar or tensor state",
            "shape and dtype are invariant across every callback application",
            "`Q` is one static type in [05-OP-36]'s scalar or recursive equality "
            "domain",
            "Repeated variables denote the same type, dtype, rank, and dimensions",
            "`tensor_scan` has [05-HOST-1]'s exact-width recurrence and adjoint",
            "`process_run` has §2.6's argv, exit-code, capture, `IO`, and outside-AD "
            "contract",
            "assertion family has [05-HOST-3] and [05-OP-35]'s equality, own-width "
            "closeness, left-to-right evaluation, zero-cotangent, and `Test` behavior",
            "No dtype-named, rank-named, evaluator-only, legacy, or compatibility "
            "identity is part of this atom",
        ),
        "05-OP-39": (
            "governs exactly `reduce_window_sum`, `reduce_window_mean`, "
            "`reduce_window_max`, `reduce_window_min`, and the internal "
            "`ReduceWindowGrad` identity",
            "Sum, max, and min admit every active numeric tensor dtype",
            "mean admits every active float tensor dtype",
            "bool, scalar, string, and reserved dtype spellings are type errors",
            "Each forward result has the input dtype and [05-RWIN-1]'s output "
            "dimensions",
            "exact valid-padding signature, reducer identity, window/stride "
            "validation",
            "per-dtype arithmetic width, no-user-accumulator rule, canonical balanced "
            "tree",
            "overflow/NaN/tie behavior, first-order adjoint, overlap accumulation, "
            "and higher-order rule",
            "Integer forms are forward-only",
            "float forms use the exact `ReduceWindowGrad` graph",
            "No target-specific rank, reducer, dtype, first-order-only, host-fallback, "
            "alias, or compatibility identity belongs to this atom",
        ),
        "05-OP-40": (
            "active\n> signed-integer or float dtype",
            "same scalar surface or on tensor\n> surfaces with identical dimensions",
            "performs selection, not arithmetic or numeric conversion",
            "first NaN in operand order",
            "exact stored bits, including payload and sign",
            "returns the first operand on every equality",
            "signed-zero equality",
            "Signed integers are compared exactly at their declared\n> width",
            "whole cotangent to the selected operand",
            "Signed-integer forms are forward-only",
            "no accumulator",
            "never lowers through arithmetic negation",
        ),
        "05-OP-41": (
            "active\n> signed-integer or float dtype",
            "direct checked\n> subtraction computes the exact mathematical difference",
            "traps\n> `Overflow` as operation `sub`",
            "never lowers through\n> `neg`",
            "[04-NUM-8]'s declared arithmetic width",
            "adjoint is `(g, neg(g))`",
            "Signed-integer forms are forward-only",
            "no accumulator",
        ),
        "05-OP-42": (
            "admits exactly one value of",
            "returns that value unchanged: the same type,",
            "the operation itself is pure and adds none",
            "the transform SHALL NOT traverse the argument's",
            "subgraph for adjoint construction or for structural "
            "differentiability",
            "reject a graph that reaches it only through this barrier",
            "shape-preserving exact zero for its",
            "`vmap` maps the identity pointwise",
            "never a mode, annotation, or effect",
            "no\n> accumulator",
        ),
        "05-OP-43": (
            "admits every active float dtype on a",
            "Its forward value is exactly the §3.3 lowering",
            "`max_elem(x, const(0.0))` under [05-OP-40]",
            "Its adjoint is its own, not [05-OP-40]'s:",
            "the input cotangent is `g` exactly where `cmplt(0, x)` is true "
            "and exact",
            "including at `x = 0`, at both signed zeros, and at",
            "remains intact through AD and every other",
            "The operation has no accumulator",
        ),
        "05-OP-44": (
            "exactly the heap-handle, strong-owner, tagged-value conversion, "
            "option-node, entry-borrow, and guarded-access callable identities",
            "typedef struct { void *handle; } chelis_string;",
            "typedef struct chelis_tensor chelis_tensor;",
            "typedef struct chelis_tensor_write chelis_tensor_write;",
            "typedef struct chelis_option chelis_option;",
            "typedef struct chelis_mapped_file chelis_mapped_file;",
            "The heap-kind universe is closed and exact: `String`, `Tensor`, "
            "`TensorStorage`, `List`, `Tuple`, `Dict`, `Adt`, `Option`, and "
            "`MappedFile`",
            "exactly one strong-owner count, and every kind has exactly one "
            "finalizer",
            "`TensorStorage` is private",
            "one retain callable, one release callable, one take conversion into a "
            "value, one take conversion out of a value, and one borrow conversion "
            "out of a value",
            "no kind-generic handle, untagged payload, owner flag, or second "
            "representation of any kind",
            "Tensor, List, tuple, dictionary, ADT, option, and mapped-file "
            "carriers are\n> pointers to incomplete C types",
            "`chelis_string` is the one fixed by-value\n> wrapper",
            "A live handle is one logical owner under [04-LIN-3]",
            "Retain creates one additional owner by a checked relaxed increment",
            "exceed the representable owner range traps `Overflow`",
            "the final release synchronizes with release/acquire ordering before "
            "running the kind's finalizer exactly once",
            "live handle whose kind disagrees with the callable or with\n> the "
            "value tag, traps `Domain`",
            "Reusing its stale pointer afterward violates the live-handle\n> "
            "precondition",
            "does not\n> promise a diagnostic or retain a tombstone",
            "there is no non-owning value",
            "`chelis_value_clone` creates one additional owner for a heap tag",
            "A take conversion out of a value moves the value's owner to the "
            "returned handle and ends the value",
            "A borrow conversion out of a value returns the handle without "
            "creating or consuming an owner",
            "A constructor clones each borrowed child exactly once",
            "a finalizer releases each stored child exactly once",
            "no value contains itself and no heap graph has a cycle",
            "`chelis_option_none` owns no child and `chelis_option_some` owns "
            "exactly one tagged child",
            "`chelis_option_unwrap` of a `None` node traps `Domain`",
            "There is no by-value, discriminant-plus-payload, or scalar-special "
            "option carrier",
            "`CHELIS_VALUE_MAPPED_FILE` is its only tagged representation",
            "A tensor handle is a descriptor that retains exactly one storage "
            "allocation for its whole lifetime",
            "`chelis_tensor_retain` and `chelis_tensor_release` are the only "
            "public tensor lifetime operations",
            "`chelis_tensor_begin_write` succeeds only when the descriptor has "
            "exactly one live owner, its storage has exactly one live descriptor, "
            "the storage is runtime-owned, and no guard is active on it",
            "non-owning guard embedded in that descriptor",
            "A successful begin invalidates\n> every read view previously returned "
            "for that descriptor before it activates\n> the guard",
            "Dereferencing one afterward violates the caller precondition; the\n> "
            "runtime does not promise to diagnose that stale pointer",
            "The guard borrows,\n> but neither consumes nor clones, the descriptor's "
            "existing owner for the\n> guard lifetime; it allocates no guard object",
            "Every other begin,\n> read view, retain, clone, or release of that "
            "descriptor traps `Domain` until\n> `chelis_tensor_end_write` consumes "
            "and deactivates the guard without freeing\n> an allocation or consuming "
            "the descriptor owner",
            "`chelis_tensor_write_view` borrows its `const` guard",
            "a compiler's reuse proof never replaces them",
            "An entry borrow, following [04-LIN-7], is a descriptor over storage "
            "the caller owns",
            "produces storage that is never runtime-owned",
            "`chelis_tensor_begin_write` on it traps `Domain`",
            "no address comparison, retain count, or later invocation makes that "
            "storage runtime-owned",
            "cannot prove a foreign allocation's lifetime or physical size",
            "no alias, wrapper, deprecated spelling, field-level access, owner "
            "flag, or free-style path",
            "no accumulator and is outside AD",
        ),
        "05-RNG-1": (
            "A random primitive is a pure function of the key it is given",
            "Every conforming evaluation produces byte-identical random results "
            "for the same key",
            "compiler version and target do not vary this result",
            "No handler, ordinal, execution order, or other state contributes to a "
            "draw",
            "A draw in an `if` or `match` arm that is not selected is not evaluated",
        ),
        "05-OBS-1": (
            "every NaN payload renders as the exact spelling `NaN`",
            "parses to §3.7's canonical quiet-NaN image at that dtype",
            "NaN text round-trips at the class level and deliberately loses "
            "payload bits",
        ),
        "05-HOST-1": (
            "A host-runtime operation SHALL preserve its complete checked signature",
            "effects, exact dtype identity, evaluation order, traps, and value result",
            "in every language execution mode",
            "SHALL NOT substitute a stub, default value, null pointer, erased dtype, "
            "or alternate helper contract",
            "Device-kernel nesting is governed by the operation's effect and "
            "device-boundary rules",
            "it does not make the host operation illegal",
        ),
        "05-HOST-2": (
            "JSON and CSV operations, `round_to`, and `process_run` are legal "
            "host-runtime operations in every language execution mode",
            "Pure parsing, projection, serialization, and rounding retain their "
            "stated purity",
            "file and process operations retain their declared `IO` effect and "
            "observable order",
            "compiled host execution SHALL produce the same typed result or language "
            "trap as evaluation",
            "device-only kernel may not perform `IO`",
            "effect-boundary fact SHALL NOT be represented as a language-wide "
            "rejection, inert stub, default value, or evaluator-only signature",
        ),
        "05-SPARSE-1": (
            "`ScatterElements` SHALL take an index tensor of any active signed-integer dtype",
            "interpreted at its exact stored width",
            "no index dtype is widened, narrowed, or otherwise converted",
            "public C gather and scatter callables in [05-OP-33] have this same "
            "complete index-dtype domain",
            "no i32/i64-only exception",
        ),
        "05-HOST-3": (
            "`test_assert` admits bool",
            "`test_assert_eq` admits exactly [05-OP-36]'s scalar and recursive "
            "equality domain",
            "`test_assert_eq_tensor` admits two same-shaped tensors of one active "
            "tensor element dtype",
            "`test_assert_close_tensor` admits two same-shaped tensors and a "
            "tolerance at one active float dtype",
            "assertion identities are generic",
            "dtype-named or rank-named aliases do not exist",
            "SHALL NOT emit an inert assertion, default value, compatibility helper, "
            "or whole-module rejection based on an unreachable assertion",
        ),
        "05-UNS-5": (
            "An unsupported diagnostic SHALL carry the authority for its disposition",
            "A language-rejected case cites the normative atom that decides it",
            "A legal operation unavailable on the selected target identifies the exact "
            "typed capability cell",
            "Those categories SHALL be distinguishable at the diagnostic surface",
            "a rejection carrying neither authority is a defect",
            "Project scheduling or issue metadata may be associated with a capability "
            "cell outside this normative contract",
            "it is not semantic authority and does not alter legality",
        ),
    }
    blocks_with_registries = dict(blocks)
    for registry_atom, registry_relative in OP_MANIFEST_REGISTRY_FILES.items():
        blocks_with_registries[registry_atom] = (
            blocks.get(registry_atom, "")
            + "\n"
            + docs.get(registry_relative, "")
        )
    for atom, requirements in atom_requirements.items():
        require_atom(blocks_with_registries, atom, requirements, violations)
    validate_op_manifests(docs, blocks, violations)

    require_all(
        spec05,
        (
            ("`InputAxis(t, a)`", "folded tensor-axis extent carrier"),
            (
                "`reshape` admits `Lit`, `Node`, `InputAxis`, and `Sym`",
                "runtime extent owner admission",
            ),
            (
                "well formed only when the start\n  paired with it is `Lit(0)`",
                "ToEnd shrink end requires a zero start",
            ),
            (
                "`expand` sets the extent at `axis` and is well formed only "
                "when the operand's\nextent at `axis` is 1",
                "expand requires a unit source extent",
            ),
            (
                "A reduction axis, `expand`'s broadcast axis, and `insert`'s"
                "\n> new-axis position SHALL be",
                "axis atom names both movement primitives",
            ),
            (
                "names the\n> dimension it creates, which is by construction not a dimension of the\n> operand; that name SHALL be statically resolvable in the same sense",
                "insert names a dimension absent from the operand",
            ),
            (
                "> `insert(g / divisor, axis, original_extent)` at the "
                "operand dtype.",
                "mean adjoint reinserts the reduced axis",
            ),
            (
                "[05-AXIS-1] governs the reduction, `expand`, and `insert`\n> family",
                "C axis family names both movement primitives",
            ),
            (
                "| `insert` | `(&tensor[D,p], axis: i32, size: i64) -> "
                "tensor[D_plus,p]` | Insert a new dimension of width `size` "
                "at position `axis`, producing rank `rank(x) + 1`.",
                "insert movement row",
            ),
            (
                "A literal operand extent at\n`axis` other than 1 is a type error. A symbolic or runtime operand extent at\n`axis` other than 1 fails that claim's runtime extent guard and traps\n`Domain`, placed and rendered per `spec/04-type-system.md` §4.7 and\n[04-NUM-9].",
                "expand non-unit source extent is rejected or traps",
            ),
            (
                "| `expand` | `insert(sum(g, axis), axis, 1i64)`",
                "expand adjoint restores the unit axis",
            ),
            (
                "| `insert` | `sum(g, axis)`",
                "insert adjoint collapses the inserted axis",
            ),
            (
                "| `cmplt(a, b)` | `cmplt(a, b)` | "
                "`and(not(nan), cmplt(a, b))` |",
                "ordered cmplt lowering",
            ),
            (
                "| `lt(a, b)` | `cmplt(a, b)` | "
                "`and(not(nan), cmplt(a, b))` |",
                "ordered lt lowering",
            ),
            (
                "| `eq(a, b)` | `not(or(lt, gt))` | "
                "`and(not(nan), not(or(lt, gt)))` |",
                "ordered eq lowering",
            ),
            (
                "| `neq(a, b)` | `or(lt, gt)` | `or(nan, or(lt, gt))` |",
                "ordered neq lowering",
            ),
            (
                "| `gt(a, b)` | `gt` | `and(not(nan), gt)` |",
                "ordered gt lowering",
            ),
            (
                "| `gte(a, b)` | `not(lt)` | `and(not(nan), not(lt))` |",
                "ordered gte lowering",
            ),
            (
                "| `lte(a, b)` | `not(gt)` | `and(not(nan), not(gt))` |",
                "ordered lte lowering",
            ),
            (
                "Here `lt = cmplt(a,b)`, `gt = cmplt(b,a)`, and, on floats only,\n"
                "`nan = or(is_nan(a),is_nan(b))`",
                "ordered comparison helper definitions",
            ),
            (
                "`print`, `to_string`, `to_list`, diagnostics",
                "to_string observation exit",
            ),
            (
                "`tensor_scan` accumulator and emitted elements remain at `T`",
                "tensor_scan exact carrier",
            ),
            (
                "The first NaN is the forward result when any NaN is\npresent",
                "window extrema first-NaN forward rule",
            ),
            (
                "including equal positive or negative infinities, so each\n  receives `g / k`",
                "window extrema infinity-tie adjoint",
            ),
            (
                "route the full `g` to the first\n  NaN in row-major window order",
                "window extrema NaN adjoint",
            ),
            (
                "Each exact [05-OP-36] identity:\n`cmplt`, `lt`, `eq`, `neq`, `gt`, "
                "`gte`, and `lte`",
                "comparison AD identity completeness",
            ),
            (
                "`cast_trunc`,\n"
                "`cast_saturate`, `cast_wrap`, `and`, `or`, `not`,\n"
                "`count`, `argmax_reduce`, and `argmin_reduce` likewise reject `grad`",
                "count structural AD rejection",
            ),
            (
                "`max_elem` routes the whole cotangent to\n"
                "the first operand when the inputs are equal",
                "max_elem first-operand tie adjoint",
            ),
            (
                "| `max_elem` | `(&tensor[D,p], &tensor[D,p]) -> tensor[D,p]` | "
                "Element-wise maximum | Whole `g` flows to the operand selected by "
                "[05-OP-40] |",
                "max_elem table tie adjoint",
            ),
            (
                "On\n> floats, it returns the first NaN in operand order when either "
                "operand is NaN",
                "max_elem first-NaN selection",
            ),
            (
                "returns the first operand on every equality, preserving its exact "
                "stored bits,\n> including signed-zero equality",
                "max_elem signed-zero tie selection",
            ),
            (
                "`relu` carries [05-OP-43]'s own adjoint instead: its "
                "gradient is exactly zero\nat `x = 0`",
                "relu zero-boundary adjoint",
            ),
            (
                "out[i] = select_max_first(a[i], b[i]);",
                "max_elem first-operand reference selection",
            ),
            (
                "illustrative\nimplementation shapes for the C backend; it is not a "
                "semantic oracle",
                "illustrative reference status",
            ),
            (
                "`sub` is a Tier 1 primitive governed by [05-OP-41]",
                "direct sub lowering",
            ),
            (
                "`min_elem` is a Tier 1 primitive governed by [05-OP-40]",
                "direct min_elem lowering",
            ),
            (
                "out[i] = select_min_first(a[i], b[i]);",
                "min_elem first-operand reference selection",
            ),
        ),
        violations,
    )



def validate_schema_and_consumers(
    docs: dict[str, str], violations: list[str]
) -> None:
    capability = docs["spec/design/capability_table.md"]
    plan = docs["spec/design/dtype_semantics.md"]
    loud = docs["spec/design/loud_unsupported.md"]
    provenance = docs["spec/design/spec_provenance.md"]
    roadmap = docs["spec/design/remediation_roadmap.md"]
    runtime_extents = docs["spec/design/runtime_extents.md"]
    runtime = docs["spec/design/runtime_representation.md"]
    surface = docs["docs/CHELIS_SURFACE.md"]
    status = docs["docs/investigations/remediation_status_2026_08_04.md"]

    require_all(
        surface,
        (
            ("introduces `IO`", "surface IO spelling"),
            ("| `IO` | file ops", "surface IO spelling"),
            (
                "dedicated `RiscOp::Relu`; forward equals stored-bit "
                "`max_elem(x, 0)`",
                "surface dedicated ReLU identity",
            ),
            (
                "`g` only where `0 < x`; exact +0 at both zeros and NaN",
                "surface ReLU adjoint",
            ),
        ),
        violations,
    )

    require_all(
        runtime,
        (
            (
                "Division is a partial exact-integer operation, not rational arithmetic",
                "runtime capacity validity domain",
            ),
            (
                "`n / n` and `1`, and `0 / n` and `0` have\n"
                "distinct keys absent such a proof",
                "runtime capacity division red controls",
            ),
            (
                "`0 * (1 / n)` retains the partial quotient and remains distinct "
                "from zero\nunless `n != 0` and `n` divides 1 have both been proved",
                "runtime zero-product validity domain",
            ),
            (
                "`CapacityKey` is deliberately carrier-independent",
                "runtime capacity carrier independence",
            ),
            (
                "The set is deliberately closed against equality learned only by "
                "passing a\n[#1277] runtime guard",
                "runtime guard capacity boundary",
            ),
            (
                "exact shrink-only transition-debt\nmanifest",
                "runtime Phase 0 shrink-only debt",
            ),
            (
                "A foreign carrier never constructs `TensorRef<T>`, `TensorMut<T>`, "
                "`&[T]`, or\n`&mut [T]`",
                "runtime foreign slice prohibition",
            ),
            (
                "[#899] closes in this phase, not Phase 1",
                "runtime representation consumer-complete exit",
            ),
            (
                "consumers complete C1's consumer-exclusivity condition in Phase "
                "4; their exact\nfrozen debt cannot grow before then",
                "runtime C1 phase boundary",
            ),
            (
                "--receipt-dir runtime-representation-receipts",
                "runtime distributed exact-head receipts",
            ),
            (
                "section-B blocker exit uses the Phase 1\nhost-capacity evidence, "
                "the Phase 3 `--host` field-seal evidence, and the landed\n"
                "[#1289]/[#1347] receipts",
                "runtime launch-gate boundary",
            ),
        ),
        violations,
    )

    require_all(
        runtime_extents,
        (
            (
                "capacity and reuse equality\nover typed extent expressions\n"
                "([`runtime_representation.md`](runtime_representation.md), [#888])",
                "runtime extent capacity boundary",
            ),
        ),
        violations,
    )

    require_all(
        capability,
        (
            ("**Status:** Phase 4B schema frozen", "Phase 4B schema status"),
            (
                "(`BuiltinId`, `SurfaceClass`, operand `Prim`, `SemanticParams`)",
                "numeric table key",
            ),
            (
                "`SurfaceClass` is exactly `Scalar | Tensor`",
                "closed SurfaceClass variants",
            ),
            (
                "Supported { signature_rule: SignatureRuleId, result_dtype_rule:",
                "typed Table-A supported cell",
            ),
            (
                "Rejected { op_atom: SpecAtomRef, diagnostic_kind: DiagnosticKind }",
                "typed semantic rejection cell",
            ),
            (
                "`eval | c-host | c-dag | hip | metal`",
                "exact backend product",
            ),
            (
                "HIP code generation implements all four widths (raw stored-bit "
                "predicates for f16/bf16)",
                "ReLU HIP completion receipt",
            ),
            (
                "Metal f64 remains the deliberate target rejection, not an "
                "unimplemented ReLU cell",
                "ReLU Metal target boundary",
            ),
            (
                "No backend cell cites [#1313] after it closes",
                "ReLU closed-issue receipt removal",
            ),
            (
                "(`BuiltinId`, `SiblingDomain`, `SiblingCaseId`, `SemanticParams`)",
                "sibling registry key",
            ),
            (
                "`SiblingDomain` is the closed enum `Container | Boundary`",
                "closed sibling domains",
            ),
            (
                "`ToStringScalar`, `ToStringTensor`, and `ToStringList`",
                "disjoint to_string cases",
            ),
            (
                "exact case enumerator is the closed set scalar, tensor, and List",
                "to_string semantic authority",
            ),
            ("[#1282] owns checker/evaluator alignment", "to_string checker owner"),
            (
                "Every sibling `Supported` row expands across\n"
                "the same exact backend set",
                "sibling backend product",
            ),
            (
                "(`ExternalCallableFamily`, `CanonicalCallableId`)**",
                "external semantic key",
            ),
            (
                "(`ExternalCallableFamily`,\n`CanonicalCallableId`, "
                "`ExternalTargetContext`)",
                "external target key",
            ),
            (
                "`RuntimeCExport -> c-runtime` and `PyO3Binding ->\n"
                "python-extension`",
                "external target contexts",
            ),
            (
                "Implemented { implementation_id: ExternalImplementationId }",
                "typed external implemented cell",
            ),
            (
                "(`CanonicalEffectRequirement`, `BackendId`)**",
                "effect disposition key",
            ),
            (
                "`Accum | IO | Test | Resource(ResourceId)`",
                "closed effect requirement domain",
            ),
            (
                "Implemented { implementation_id: EffectImplementationId }",
                "typed effect implemented cell",
            ),
            (
                "There is no\nmissing-row, wildcard, or default disposition.",
                "effect no-default rule",
            ),
            (
                "sealed `CompleteEffectDependencies`",
                "completed effect dependency carrier",
            ),
            (
                "explicit `Pure` case",
                "explicit effect purity result",
            ),
            (
                "An exported stdlib definition has no independent external target cell",
                "derived stdlib execution rule",
            ),
            (
                "Unimplemented { issue: #2339, diagnostic_kind: UnsupportedFeature }",
                "device reduction implementation owner",
            ),
            (
                "Unimplemented { issue: #1290, diagnostic_kind: UnsupportedFeature }",
                "product implementation owner",
            ),
            (
                "Unimplemented { issue: #2266, diagnostic_kind: UnsupportedFeature }",
                "logical implementation owner",
            ),
            (
                "arithmetic reductions x (any) x bool",
                "bool reduction rejection",
            ),
            ("[05-OP-29], [#1287], [#1291]", "count capability owner"),
            (
                "`max_elem`/`min_elem` x Scalar/Tensor x active numeric dtypes "
                "([05-OP-40], [#715], [#1306], [#2338])",
                "extrema capability owner",
            ),
            (
                "`sub` x Scalar/Tensor x active numeric dtypes "
                "([05-OP-41], [#1306], [#2338])",
                "sub capability owner",
            ),
            (
                "Unimplemented { issue: #2338, diagnostic_kind: "
                "UnsupportedFeature }",
                "direct arithmetic implementation owner",
            ),
            (
                "They do not prescribe an explicit cast or arithmetic lowering",
                "no cast counting compatibility",
            ),
            (
                "axis values are not table axes",
                "count axes stay in the signature rule",
            ),
            (
                "runtime-axis `shape`, `ReduceWindow`, and `ReduceWindowGrad` "
                "([05-OP-7], [05-SHAPE-1], [05-OP-39], [#1298])",
                "runtime-axis and window capability owner",
            ),
            (
                "no host fallback, permanent target rejection, zero adjoint, or "
                "signature narrowing is an implementation receipt",
                "runtime-axis target-independent signature",
            ),
            (
                "host numeric builtins `tensor_scan`, `process_run`, and generic "
                "assertions ([05-OP-38], [#1297])",
                "host numeric capability owner",
            ),
            (
                "Unit, tuple, `Dict`, `Option`, ADT, function, deferred, and "
                "resource cases are semantic `Rejected` rows",
                "to_string rejected cases",
            ),
            (
                "The domain and case declarations\n  themselves are [#1294] "
                "prerequisite artifacts",
                "capability pre-4C builtin declarations",
            ),
            ("invokes the 4B, 4C, and 4D oracles", "nested Phase 4B oracle"),
        ),
        violations,
    )
    for stale in ("NumericSurface", "open question 5", "`to_string` x Tensor/List"):
        if stale in capability:
            violations.append(f"stale capability-schema text remains: {stale}")

    require_all(
        plan,
        (
            (
                "### Phase 4B - decided-contract and schema freeze (this change)",
                "Phase 4B plan",
            ),
            (
                ".venv/bin/python scripts/dtype_phase4b_oracle.py",
                "Phase 4B oracle command",
            ),
            (PASS_LINE, "Phase 4B oracle success line"),
            (
                "The additive-contradiction leg is an acknowledgement, not a "
                "whole-file digest.",
                "Phase 4B acknowledgement gate replaces whole-file digests",
            ),
            (
                "requires each changed file to be named in the pull request "
                "body",
                "Phase 4B acknowledgement is per changed file",
            ),
            (
                "An unacknowledged\nchange and an acknowledgement naming an "
                "unchanged file both fail\n`--require-acknowledgement`, which "
                "the dedicated `PR Contract Acknowledgements`\ncheck applies "
                "through `phase4b_change_report.py`. The full Phase 4B oracle "
                "runs\nindependently in Docs.",
                "Phase 4B acknowledgement enforcing mode",
            ),
            (
                ".venv/bin/python scripts/dtype_count_oracle.py",
                "Count child oracle command",
            ),
            ("DTYPE COUNT ORACLE: PASS", "Count child oracle success line"),
            (
                "checker grammar, dedicated non-alias\n`Count` IR",
                "Count child oracle ownership",
            ),
            (
                ".venv/bin/python\nscripts/dtype_relu_oracle.py",
                "ReLU child oracle command",
            ),
            ("DTYPE RELU ORACLE:\nPASS", "ReLU child oracle success line"),
            (
                "f16/bf16/f32/f64 raw-bit inputs and outputs",
                "ReLU HIP execution width matrix",
            ),
            (
                "(ExternalCallableFamily, CanonicalCallableId, "
                "ExternalTargetContext)",
                "dtype-plan external target key",
            ),
            (
                "exported-stdlib dependency derivation",
                "dtype-plan stdlib derivation",
            ),
            (
                "(CanonicalEffectRequirement, BackendId)",
                "dtype-plan effect key",
            ),
            (
                "CompleteEffectDependencies::Pure",
                "dtype-plan explicit effect purity",
            ),
            ("invokes the 4B, 4C, and 4D oracles", "dtype-plan final nesting"),
            ("[#1290] balanced sum/product backend work", "dtype-plan product owner"),
            ("[#1281] mean/extrema/argument-reduction", "dtype-plan reduction owner"),
            ("[#1284]\n   owns replacing", "dtype-plan logical owner"),
            (
                "scalar/tensor/recursive-List `to_string`; boolean\n   "
                "`and`/`or`/`not`; NaN-aware "
                "comparison/equality; exact scalar/container/C\n   tensor carriers",
                "dtype-plan to_string semantics",
            ),
            (
                "byte-exact recursive runtime List/tuple/Dict/ADT\n   observation",
                "dtype-plan recursive runtime rendering",
            ),
            (
                "every active signed-integer sparse-index width across IR and\n"
                "   public C",
                "dtype-plan sparse index domain",
            ),
            (
                "canonical gradient consumer-edge order by forward node ordinal\n"
                "   and input slot",
                "dtype-plan gradient edge order",
            ),
            (
                "legal compiled host\n   effects with the exact language spelling `IO`",
                "dtype-plan IO spelling",
            ),
            (
                "This is the executable requirement for zero capacity exceptions.",
                "zero capacity exceptions",
            ),
            (
                "[#1282] [05-OP-25] scalar/tensor/recursive-List `to_string` domain",
                "dtype-plan to_string checker owner",
            ),
            (
                "[#1059] compiled C-host Tensor/List rendering cells",
                "dtype-plan to_string backend owner",
            ),
            (
                "### Pre-4C - exact builtin-atom closure ([#1294])",
                "pre-4C atom-closure gate",
            ),
            (
                "Before any Phase 4C key/cell type, authoring macro, or partial "
                "machine row may\nland",
                "atom closure precedes every partial Phase 4C mechanism",
            ),
            (
                "[#1294] first introduces the closed builtin domain/case declaration "
                "types and\nattaches a non-empty exhaustive declaration to every "
                "`BuiltinDecl`",
                "pre-4C builtin domain declarations",
            ),
            (
                "discovers the union of every canonical Table-A IR/RISC operation "
                "identity and\nevery declared `BuiltinDecl` sibling domain/case and "
                "proves an exact\nbijection from every Table-A and sibling-builtin "
                "identity to one\nsemantically governing normative `[05-OP-N]` line",
                "total exact builtin-atom bijection",
            ),
            (
                "authors every missing\natom, regenerates the rejection registry",
                "atom closure authors and regenerates",
            ),
            (
                "admits no count allowlist,\nunnumbered table/prose authority, issue "
                "citation, default, alias, age, or\ncompatibility exception",
                "atom closure has zero authority exceptions",
            ),
            (
                "An open implementation issue can authorize only a\nlater Table-B "
                "`Unimplemented` receipt; it never satisfies semantic closure",
                "implementation issues are Table-B-only",
            ),
            (
                ".venv/bin/python scripts/dtype_builtin_atom_closure_oracle.py",
                "builtin atom-closure oracle command",
            ),
            (
                "DTYPE BUILTIN ATOM CLOSURE ORACLE: PASS",
                "builtin atom-closure oracle success line",
            ),
            (
                "The oracle and its\nadversarial mutations must be green and merged "
                "before Phase 4C begins",
                "atom-closure merge gate",
            ),
            (
                "### Pre-4C - composite executable gate ([#1296])",
                "pre-4C composite gate",
            ),
            (
                "After the individual behavior, storage, census, and [#1294] "
                "atom-closure\noracles land",
                "composite gate follows all prerequisite oracles",
            ),
            (
                ".venv/bin/python scripts/dtype_pre_phase4c_oracle.py",
                "composite pre-4C oracle command",
            ),
            (
                "DTYPE PRE-PHASE-4C ORACLE: PASS",
                "composite pre-4C oracle success line",
            ),
            (
                "fails for a missing, duplicate,\nskipped, stale, nonzero, or "
                "success-line-free leg",
                "composite pre-4C runner totality",
            ),
            (
                "It is wired to the normal gate; prose coverage\nor a manual waiver "
                "is not an entry receipt",
                "composite pre-4C normal-gate wiring",
            ),
            (
                "**Entry condition:** the [#1296] composite pre-4C oracle is green, "
                "merged, and\nwired to the normal gate",
                "Phase 4C composite entry condition",
            ),
            (
                "It includes [#1294]'s exact builtin-atom closure",
                "Phase 4C entry includes exact atom closure",
            ),
            (
                "[#1284], [#1287]-[#1298], and [#1306]",
                "composite gate includes every late prerequisite",
            ),
            (
                "chelis#1295's all-active-float random/rounding rules, chelis#1297's "
                "compiled\nhost effects, chelis#1298's runtime-axis/window operations, "
                "and chelis#1306's\ndirect subtraction/extrema identities have landed",
                "Phase 4C issue-owned behavior prerequisites",
            ),
            (
                "[#1306] direct checked subtraction and stored-bit extrema selection",
                "dtype-plan direct arithmetic owner",
            ),
        ),
        violations,
    )
    require_all(
        loud,
        (
            (
                "consume `chelis_vocab::EffectKind` directly; no "
                "`chelis_types` re-export",
                "EffectKind direct ownership",
            ),
            ("`chelis_scalar { chelis_dtype dtype; uint64_t bits; }`", "C scalar carrier"),
            ("`chelis_tensor` pairs `void *data` with `chelis_dtype`", "C tensor carrier"),
            ("rank and positional axes are `int32_t`", "C axis domain"),
            ("require compiled-artifact ABI version 2", "artifact ABI v2"),
            (
                "Under [05-OP-32], dictionary keys are exactly `string`, `bool`, or "
                "any active\nsigned-integer scalar dtype",
                "dict key domain",
            ),
            ("No compatibility bridge is permitted", "no host compatibility bridge"),
            (
                "BuiltinId × SiblingDomain × SiblingCaseId × SemanticParams",
                "loud sibling key",
            ),
            (
                "(ExternalCallableFamily, CanonicalCallableId, "
                "ExternalTargetContext)",
                "loud external target key",
            ),
            (
                "generated transitive\n  dependency closure over the checked body",
                "loud stdlib derivation",
            ),
            (
                "(CanonicalEffectRequirement, BackendId)",
                "loud effect key",
            ),
            (
                "CompleteEffectDependencies::Pure",
                "loud explicit effect purity",
            ),
            ("also invokes the Phase 4B oracle", "loud final-oracle nesting"),
        ),
        violations,
    )
    require_all(
        provenance,
        (
            (
                "ExternalCallableFamily × CanonicalCallableId × "
                "ExternalTargetContext",
                "provenance external target key",
            ),
            (
                "generated transitive dependency closure over the checked body",
                "provenance stdlib derivation",
            ),
            (
                "external target dispositions, exact effect dispositions, and "
                "derived exported-stdlib dependencies",
                "provenance governed-surface summary",
            ),
            (
                "exact effect dispositions",
                "provenance effect authority",
            ),
        ),
        violations,
    )
    require_all(
        roadmap,
        (
            (
                "[`runtime_representation.md`](runtime_representation.md) ([#893])",
                "runtime representation owner",
            ),
            (
                "Its current children are [#899], [#889], [#1289], [#1345], "
                "[#1360], and [#1364]",
                "runtime representation child graph",
            ),
            (
                "[#888] is an explicitly linked Phase 1 interlock and [#1347] a "
                "landed zero-extent receipt",
                "runtime representation interlocks",
            ),
            ("[#1290] replaces noncanonical product/sum trees", "roadmap product owner"),
            ("[#1281] owns the remaining reduction rows", "roadmap reduction owner"),
            ("Phase 4B froze semantics", "roadmap Phase 4B boundary"),
            (
                "canonical scalar/tensor/recursive-List `to_string` and compiled "
                "C-host Tensor/List rendering "
                "([#1282]/[#1059])",
                "roadmap to_string checker owner",
            ),
            (
                "canonical scalar/tensor/recursive-List `to_string` and compiled "
                "C-host Tensor/List rendering "
                "([#1282]/[#1059])",
                "roadmap to_string backend owner",
            ),
            (
                "external-target/effect dispositions",
                "roadmap external target authority",
            ),
            ("effect dispositions", "roadmap effect authority"),
            (
                "full comparison/equality [05-OP-36] and typed logical/`where` "
                "lowering ([#1284])",
                "roadmap logical owner",
            ),
            ("first-class `count` [05-OP-29]", "roadmap count contract"),
            ("zero-exception census ([#1288])", "roadmap census prerequisite"),
            (
                "[#1286]'s verified compiled ownership and opaque unified-heap ABI "
                "through [#1362]'s C-lane `--phase launch` oracle",
                "roadmap launch ownership gate",
            ),
            (
                "full [#1286] class closure, including HIP and non-launch children, "
                "remains tracker work and does not gate v0.19",
                "roadmap full ownership boundary",
            ),
            ("WireDag v6 exact-only break", "roadmap wire v6 break"),
            (
                "[#1295] owns all-active-float rounding/random parameter contracts "
                "and all-dtype padding",
                "roadmap random owner",
            ),
            ("[#1297] owns legal compiled host-effect execution", "roadmap host-effect owner"),
            (
                "[#1298] owns runtime-axis shape and complete window cells",
                "roadmap runtime-axis owner",
            ),
            (
                "direct stored-bit extrema plus direct checked subtraction "
                "[05-OP-40..41] ([#1306])",
                "roadmap direct arithmetic owner",
            ),
            (
                "[#1306]'s direct arithmetic/extrema oracle",
                "roadmap direct arithmetic prerequisite",
            ),
            (
                "exported-stdlib dependency closure",
                "roadmap stdlib derivation",
            ),
        ),
        violations,
    )
    require_all(
        status,
        (
            (
                "this revision is `main` at `4e200061`",
                "status reviewed execution basis",
            ),
            ("`main` is eight commits ahead", "status release distance"),
            (
                "refreshed\nagainst `4e200061` wherever this revision changes it",
                "status refreshed execution basis",
            ),
            (
                "describes `main` at\n`4e200061`",
                "status live-inventory execution basis",
            ),
            ("This change is Phase 4B", "status Phase 4B statement"),
            ("[05-OP-1..41]", "status Phase 4B atom range"),
            (
                "direct stored-bit extrema selection and checked subtraction",
                "status direct arithmetic semantics",
            ),
            ("canonical balanced sum/product/count tree", "status balanced tree"),
            (
                "canonical gradient consumer-edge order by forward node ordinal "
                "and input slot",
                "status gradient edge order",
            ),
            (
                "byte-exact runtime List/tuple/Dict/ADT observation",
                "status recursive runtime rendering",
            ),
            (
                "all-active-signed sparse indices across IR and public C",
                "status sparse index domain",
            ),
            ("exact `IO` effect spelling", "status IO spelling"),
            ("#893/#1289 seal tensor access", "status carrier prerequisite"),
            (
                "#1288 replaces every frozen capacity disposition",
                "status census prerequisite",
            ),
            ("#1287 lands `count` in eval/C", "status count owner"),
            ("#1291 owns loud HIP/Metal count cells", "status count GPU owner"),
            ("#1295", "status random and padding owner"),
            ("#1296", "status composite gate owner"),
            ("#1297", "status host-effect owner"),
            ("#1298", "status runtime-axis owner"),
            ("#1306", "status direct arithmetic owner"),
            ("No compatibility wrapper, reader", "status no compatibility"),
            ("Of the 139 issues parented", "status parented count"),
            ("54 remain open", "status open count"),
            ("#729 | 27 / 65", "status #729 count"),
            (
                "#1293 aligns all 83 recursively discovered stdlib numeric "
                "definitions",
                "status stdlib alignment owner",
            ),
            (
                "external target dispositions, and exact effect-disposition rows",
                "status Phase 4C external target delivery",
            ),
            (
                "exact effect-disposition rows",
                "status Phase 4C effect delivery",
            ),
            (
                "exported-stdlib dependency closures",
                "status Phase 4D stdlib derivation",
            ),
            (
                "total external target-disposition registry",
                "status external target authority",
            ),
            (
                "derive per-backend executability transitively from their checked\n"
                "   bodies",
                "status stdlib derivation",
            ),
        ),
        violations,
    )

    tracker_counts = [
        (int(open_count), int(total_count))
        for open_count, total_count in re.findall(
            r"^\| #7\d+(?: \(closed\))? \| (\d+) / (\d+)",
            status,
            re.MULTILINE,
        )
    ]
    if len(tracker_counts) != 5:
        violations.append("status issue graph must contain exactly five tracker rows")
    elif sum(open_count for open_count, _total in tracker_counts) != 54:
        violations.append("status issue graph tracker rows must sum to 54 open issues")
    elif sum(total for _open_count, total in tracker_counts) != 139:
        violations.append("status issue graph tracker rows must sum to 139 total issues")


def validate_contract(root: Path = REPO_ROOT) -> None:
    docs = {relative: read(root, relative) for relative in CONTRACT_FILES}
    violations: list[str] = []
    validate_normative_contract(docs, violations)
    validate_schema_and_consumers(docs, violations)
    validate_required_contract(docs, violations)
    if violations:
        raise OracleError("; ".join(violations))


def run_oracle(python: str = sys.executable, root: Path = REPO_ROOT) -> None:
    """Run the content leg: the normative, schema, and digest contracts.

    `main` runs the acknowledgement leg before this one, so the success line is
    never reached with an unacknowledged frozen contract change.
    """

    try:
        validate_contract(root)
        subprocess.run(
            (python, "scripts/generate_rejection_registries.py", "--check"),
            cwd=root,
            check=True,
        )
    except OracleError as error:
        raise SystemExit(f"DTYPE PHASE 4B ORACLE: FAIL: {error}") from error
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            "DTYPE PHASE 4B ORACLE: FAIL: rejection registry disagreement "
            f"(exit {error.returncode})"
        ) from error
    print(PASS_LINE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "chelis#729 Phase 4B freeze oracle. Validates the normative "
            "contract and checks that every changed frozen contract file is "
            "acknowledged by name."
        )
    )
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument(
        "--base",
        default=DEFAULT_BASE_REF,
        help=(
            "base ref for the frozen contract diff (default: "
            f"{DEFAULT_BASE_REF}). The comparison point is the merge base of "
            "this ref and HEAD, so a change already on the base branch is "
            "never reported as this branch's."
        ),
    )
    comparison.add_argument(
        "--pr-head", default=None,
        help="exact PR event head; validate the synthetic merge and use its first parent",
    )
    parser.add_argument(
        "--acknowledge",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "acknowledge one changed file path, atom:<ID>, or region:<JSON string>; "
            "repeatable. The local equivalent of a pull request body line."
        ),
    )
    parser.add_argument(
        "--acknowledgements-file",
        metavar="FILE",
        help=(
            "read acknowledgement lines from FILE (a saved pull request body). "
            f"'-' reads stdin. Each line is '{ACKNOWLEDGEMENT_KEY} <address>'."
        ),
    )
    parser.add_argument(
        "--acknowledgements-env",
        metavar="NAME",
        help=(
            "read acknowledgement lines from the environment variable NAME. "
            "CI uses this so an untrusted pull request body never reaches a "
            "shell command line."
        ),
    )
    parser.add_argument(
        "--require-acknowledgement",
        action="store_true",
        help=(
            "fail on an unacknowledged contract change, a stale or malformed "
            "acknowledgement, or an unresolvable merge base. This is the local "
            "full-oracle equivalent of the dedicated PR acknowledgement "
            "check; without it the acknowledgement leg reports and exits 0."
        ),
    )
    return parser


def acknowledgement_body(args: argparse.Namespace) -> str | None:
    """Return the acknowledgement document named by the parsed arguments."""

    sources = [args.acknowledgements_file, args.acknowledgements_env]
    if all(source is None for source in sources):
        return None
    if all(source is not None for source in sources):
        raise SystemExit(
            "DTYPE PHASE 4B ORACLE: FAIL: pass at most one of "
            "--acknowledgements-file and --acknowledgements-env"
        )
    if args.acknowledgements_env is not None:
        name = args.acknowledgements_env
        if name not in os.environ:
            raise SystemExit(
                "DTYPE PHASE 4B ORACLE: FAIL: acknowledgement environment "
                f"variable {name} is not set"
            )
        return os.environ[name]
    if args.acknowledgements_file == "-":
        return sys.stdin.read()
    try:
        return Path(args.acknowledgements_file).read_text(encoding="utf-8")
    except OSError as error:
        raise SystemExit(
            "DTYPE PHASE 4B ORACLE: FAIL: cannot read acknowledgements file "
            f"{args.acknowledgements_file}: {error}"
        ) from error


def main(argv: list[str] | None = None, root: Path = REPO_ROOT) -> None:
    args = build_parser().parse_args(argv)
    body = acknowledgement_body(args)
    try:
        values, errors = parse_acknowledgements(body or "")
        identities = []
        for value in [*values, *args.acknowledge]:
            try:
                identities.append(acknowledgement_identity(value))
            except OracleError as error:
                errors.append(str(error))
        if errors and args.require_acknowledgement:
            raise OracleError("; ".join(errors))
        base = args.base
        if args.pr_head is not None:
            base, _ = resolve_comparison(root, "HEAD", "", args.pr_head)
        changes = None
        comparison_error = None
        try:
            changes = changed_contracts(root, base)
        except (ValueError, OSError) as error:
            comparison_error = str(error)
        report = validate_frozen_contract_changes(
            root=root,
            base=base,
            acknowledgements=tuple(identity for kind, identity in identities if kind == "file"),
            require_acknowledgement=args.require_acknowledgement,
            contract_files=CONTRACT_FILES,
            committed_contract_changes=tuple(changes["changed_contract_files"]) if changes is not None else (),
        )
        report.extend(f"  ISSUE {error}" for error in errors)
        if comparison_error is not None:
            if args.require_acknowledgement:
                raise OracleError(comparison_error)
            report.append(f"identity acknowledgement: comparison unavailable (advisory): {comparison_error}")
        if changes is not None:
            named = [(kind, identity) for kind, identity in identities if kind != "file"]
            violations = identity_acknowledgement_violations(changes, named)
            if violations and args.require_acknowledgement:
                raise OracleError("; ".join(violations))
            for row in changes["changes"]:
                key = (row["kind"], row["identity"])
                marker = "ok " if key in named else "NEEDS"
                report.append(f"  {marker} {acknowledgement_line(*key)}")
            report.extend(f"  ISSUE {violation}" for violation in violations)
    except (OracleError, ValueError) as error:
        raise SystemExit(f"DTYPE PHASE 4B ORACLE: FAIL: {error}") from error
    for line in report:
        print(line)
    sys.stdout.flush()
    run_oracle(sys.executable, root)


if __name__ == "__main__":
    main()
