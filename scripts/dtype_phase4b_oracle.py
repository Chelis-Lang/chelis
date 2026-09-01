#!/usr/bin/env python3
"""Authoritative chelis#729 Phase 4B semantic/schema freeze oracle.

Acceptance is exit 0 with the final line ``DTYPE PHASE 4B ORACLE: PASS``.

Usage:

    .venv/bin/python scripts/dtype_phase4b_oracle.py
"""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import re
import subprocess
import sys


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
    "openspec/specs/risc-primitives/spec.md",
    "openspec/specs/serialization/spec.md",
    "openspec/specs/transformations/spec.md",
    "openspec/specs/type-system/spec.md",
    "spec/registry/c_scalar_carrier.md",
    "spec/registry/c_container_boundary.md",
    "spec/registry/c_tensor_runtime.md",
    "spec/registry/stdlib_adt_identities.md",
    "spec/registry/stdlib_numeric_manifest.md",
)
FROZEN_FILE_DIGESTS = {
    "spec/02-surf-syntax.md": (
        "7eb644de9c58a57400f588c0f411123096013448e8f6f6e088fb1972d14f9b48"
    ),
    "spec/03-deep-syntax.md": (
        "efb29e77a338c94bb6c6c6b9adc6c4bb236f5d78c0c329f3775a0b48b8640348"
    ),
    "spec/04-type-system.md": (
        "21b3b2f3c8aafc52c975e86e1bf573a327f33e1be845f9093382fd61c45b3045"
    ),
    "spec/05-risc-primitives.md": (
        "e427fa744b313a3a8a7cad99e1ca92ed7df8e57db459bf53a499edb098dca86a"
    ),
    "spec/06-transformations.md": (
        "30a78217103d0a57d63b9e1a3d927d2e1079affa4d1d0641ef36a9afff34e25a"
    ),
    "spec/10-serialization.md": (
        "58f707d4e155d098962db224317061684b2c026816cab234ba026d560510a6da"
    ),
    "spec/11-ffi.md": (
        "0d0b5065f90ae704ecea26744208e76f1075907f8ea49279a1a958db6f1c14b3"
    ),
    "spec/design/capability_table.md": (
        "b0de5579fa3b7c076ff5e809b2b6d8accf741d82305fe198233cce64ddefaf1c"
    ),
    "spec/design/compiled_value_ownership.md": (
        "4a6013c2dc5a074d19ae12696c109a35218facfdb423d29ff07cef79fe6689b9"
    ),
    "spec/design/dtype_semantics.md": (
        "61092ac8978986d2819febf6add1cebc9f11f48a26e151d09c389646d89dc6bb"
    ),
    "spec/design/implicit_linearity.md": (
        "f03302f4b328841d79824f6326f1edf2e954a9b98c0118992d4a3a1f2dfb67cf"
    ),
    "spec/design/loud_unsupported.md": (
        "6c8c5dec977fb044ced99da0f424a2c3782cca404e7884239cd044567c551f71"
    ),
    "spec/design/spec_provenance.md": (
        "6e206f634ce6062d56701f0dea0bf57bcbdca4fbf630a6c12264a904f14ea426"
    ),
    "openspec/specs/risc-primitives/spec.md": (
        "875f1094b4fba6842f90fc96c17aa37c6c2be7a5a40957b0140a10992be601a6"
    ),
    "openspec/specs/serialization/spec.md": (
        "ef0139de7e1da5ec986ec5ec4bfb12710a5cee8e77e9c91840d478404907b5ed"
    ),
    "openspec/specs/transformations/spec.md": (
        "a927fa0540c9bbb03a24fb752838409980af83d918f8f6bc8498b32ea7ab0e6f"
    ),
    "openspec/specs/type-system/spec.md": (
        "135fd5d18b3bbbffa851720973984e3b61ed18ee0b83fa8884eb7e728b38c811"
    ),
    "spec/registry/c_scalar_carrier.md": (
        "2e7b6a27b84e8c71d179b0ee10cd6c44651c4fcf66bb13dd56f0476245ffd22e"
    ),
    "spec/registry/c_container_boundary.md": (
        "4a462ef8b452b5d744e8f8207d69cffd10512cd178c759a6c112f2600a6e56f5"
    ),
    "spec/registry/c_tensor_runtime.md": (
        "cc57955f030ad18616333d6e1048e8607e54e9fd9c907bcd0698e6345b621f15"
    ),
    "spec/registry/stdlib_adt_identities.md": (
        "59c8654819ffd315028f68a91e049a5f603464f4511ae604c3634e79f8667ce5"
    ),
    "spec/registry/stdlib_numeric_manifest.md": (
        "2d6286daca1e7b16b79fcc863190d030c1291d54236b13f64a0a065090f22438"
    ),
}
OP_ATOM = re.compile(r"^> \*\*\[05-OP-(\d+)\]\*\*", re.MULTILINE)
ATOM_START = re.compile(
    r"^> \*\*\[(\d{2}-[A-Z]+-\d+)\]\*\*", re.MULTILINE
)
EXPECTED_PHASE4B_OP_HEADINGS = {
    1: "`round_to(x, places) -> r`",
    2: "Ingestion preserves",
    3: "`io/json::json_int`",
    4: "`JsonFloat(value)`",
    5: "`io/json::to_json`",
    6: "`cast_trunc(source, target)`",
    7: "The runtime extent read",
    8: "`uniform_like(template, low, high) -> result`",
    9: "`pad_sequences(sequences: List[List[T]], pad: T) ->",
    10: "`pad_sequences_to(sequences: List[List[T]], width: int64,",
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
    37: "`dropout(input, rate) -> result`",
    38: "`host_numeric_builtin(arguments...) -> result`",
    39: "`window_reduction(arguments...) -> result`",
    40: "`max_elem(left, right) -> result` and",
    41: "`sub(left, right) -> result`",
    42: "`stop_gradient(value) -> result`",
    43: "`relu(x) -> result`",
}

# These are independent, executable copies of the exact normative manifests.
# The full-file and atom digests below make additive prose tamper-evident, while
# these rows make a missing, renamed, duplicated, or retyped callable explain
# itself as a manifest failure rather than only as an opaque hash mismatch.
EXPECTED_OP_MANIFESTS = {
    "05-OP-31": tuple(
        """\
| dtype storage size | `int64_t chelis_dtype_size(chelis_dtype dtype)` |
| scalar validation/construction | `chelis_scalar chelis_scalar_from_bits(chelis_dtype dtype, uint64_t bits)` |
| value boxing | `chelis_value chelis_value_from_scalar(chelis_scalar value)` |
| value extraction | `chelis_scalar chelis_value_as_scalar(chelis_value value)` |
| rank-zero tensor construction | `chelis_tensor *chelis_scalar_tensor(chelis_scalar value)` |
| rank-zero tensor extraction | `chelis_scalar chelis_tensor_to_scalar(const chelis_tensor *tensor)` |
| tensor fill | `void chelis_fill_scalar(chelis_tensor *tensor, chelis_scalar value)` |
| scalar rendering | `chelis_string chelis_string_from_scalar(chelis_scalar value)` |
| scalar parsing | `chelis_option_scalar chelis_parse_scalar(chelis_string text, chelis_dtype dtype)` |
| exact dictionary scalar lookup | `chelis_option_scalar chelis_dict_get_scalar(const chelis_dict *dict, chelis_value key, chelis_dtype dtype)` |""".splitlines()
    ),
    "05-OP-32": tuple(
        """\
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
| dictionary lookup | `chelis_option_value chelis_dict_get(const chelis_dict *dict, chelis_value key)` |
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
| borrowed view | `chelis_tensor *chelis_alloc_view(int32_t rank, const int64_t *shape, chelis_dtype dtype, void *data, int64_t byte_capacity)` |
| rank | `int32_t chelis_tensor_rank(const chelis_tensor *tensor)` |
| extent | `int64_t chelis_tensor_shape(const chelis_tensor *tensor, int32_t axis)` |
| element count | `int64_t chelis_tensor_numel(const chelis_tensor *tensor)` |
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
| contraction | `chelis_tensor *chelis_tensor_einsum(chelis_string equation, const chelis_tensor *left, const chelis_tensor *right, chelis_dtype accumulator)` |""".splitlines()
    ),
    "05-OP-34": tuple(
        """\
| `io/json::Json` | `JsonNull | JsonBool(bool) | JsonInt(int64) | JsonBigInt(string) | JsonFloat(f64) | JsonString(string) | JsonArray(List[Json]) | JsonObject(Dict[string,Json])` |
| `decimal::Decimal` | `Decimal { coefficient: int64, scale: int64 }` |
| `time::Date` | `Date { year: int64, month: int64, day: int64 }` |
| `time::Duration` | `Duration { days: int64, hours: int64, minutes: int64, seconds: int64 }` |
| `tokenizer::Tokenizer` | `BpeTokenizer(Dict[string,int64], Dict[string,int64], Dict[int64,string], int64)` |""".splitlines()
    ),
    "05-OP-35": tuple(
        """\
| `contracts::normal_cdf` | `(p_float)->p_float` |
| `contracts::normal_cdf_contract_samples` | `()->int64` |
| `contracts::normal_cdf_contract_seed` | `()->int64` |
| `contracts::standard_contract_tolerance` | `()->f32` |
| `decimal::decimal` | `(string)->Decimal` |
| `decimal::decimal_add` | `(Decimal,Decimal)->Decimal` |
| `decimal::decimal_div` | `(Decimal,Decimal,int64,RoundingMode)->Decimal` |
| `decimal::decimal_eq` | `(Decimal,Decimal)->bool` |
| `decimal::decimal_from_int` | `(int64)->Decimal` |
| `decimal::decimal_gt` | `(Decimal,Decimal)->bool` |
| `decimal::decimal_gte` | `(Decimal,Decimal)->bool` |
| `decimal::decimal_lt` | `(Decimal,Decimal)->bool` |
| `decimal::decimal_lte` | `(Decimal,Decimal)->bool` |
| `decimal::decimal_mul` | `(Decimal,Decimal)->Decimal` |
| `decimal::decimal_sub` | `(Decimal,Decimal)->Decimal` |
| `decimal::decimal_to_float` | `(Decimal)->f64` |
| `decimal::decimal_to_string` | `(Decimal)->string` |
| `decimal::try_decimal` | `(string)->Option[Decimal]` |
| `index::drop_list` | `(List[T],int64)->List[T]` |
| `index::list_index` | `(List[T],int64)->T` |
| `index::take_list` | `(List[T],int64)->List[T]` |
| `init/kaiming::kaiming_normal` | `(&tensor[..r,p_float],p_float)->tensor[..r,p_float]!{Random}` |
| `init/kaiming::kaiming_uniform` | `(&tensor[..r,p_float],p_float)->tensor[..r,p_float]!{Random}` |
| `init/random::normal_like` | `(&tensor[..r,p_float],p_float,p_float)->tensor[..r,p_float]!{Random}` |
| `init/xavierext::trunc_normal` | `(&tensor[..r,p_float],p_float,p_float,p_float,p_float)->tensor[..r,p_float]!{Random}` |
| `init/xavierext::xavier_normal` | `(&tensor[..r,p_float],p_float,p_float)->tensor[..r,p_float]!{Random}` |
| `init/xavierext::xavier_uniform` | `(&tensor[..r,p_float],p_float,p_float)->tensor[..r,p_float]!{Random}` |
| `io/json::json_array` | `(Option[Json])->Option[List[Json]]` |
| `io/json::json_bigint` | `(Option[Json])->Option[string]` |
| `io/json::json_bool` | `(Option[Json])->Option[bool]` |
| `io/json::json_float` | `(Option[Json])->Option[f64]` |
| `io/json::json_get` | `(Json,string)->Option[Json]` |
| `io/json::json_int` | `(Option[Json])->Option[int64]` |
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
| `io::mmap_size` | `(string)->int64!{IO}` |
| `io::read_head_bytes` | `(string,int64)->List[int64]!{IO}` |
| `process::run` | `(string,List[string])->(int64,string,string)!{IO}` |
| `process::run_chelis` | `(List[string])->(int64,string,string)!{IO}` |
| `scalar::abs` | `(p_numeric)->p_numeric` |
| `scalar::max` | `(p_numeric,p_numeric)->p_numeric` |
| `scalar::min` | `(p_numeric,p_numeric)->p_numeric` |
| `sort::sort` | `(&tensor[..r,p_numeric],int32)->(tensor[..r,p_numeric],tensor[..r,int64])` |
| `tensor/construct::arange` | `(p_int,p_int)->tensor[n,p_int]` |
| `tensor/construct::linspace` | `(p_float,p_float,int64)->tensor[n,p_float]` |
| `tensor/construct::squeeze` | `(&tensor[..pre,1,..post,p],int32)->tensor[..pre,..post,p]` |
| `tensor/construct::stack` | `(List[tensor[..pre,..post,p]],int32)->tensor[..pre,rows,..post,p]` |
| `tensor/construct::unsqueeze` | `(&tensor[..pre,..post,p],int32)->tensor[..pre,1,..post,p]` |
| `tensor/mask::where_indices` | `(&tensor[..r,bool])->tensor[hits,int64]` |
| `test::assert_close` | `(p_float,p_float,p_float,string)->unit!{Test}` |
| `test::assert_close_tensor` | `(&tensor[..r,p_float],&tensor[..r,p_float],p_float,string)->unit!{Test}` |
| `test::assert_eq` | `(Q,Q,string)->unit!{Test}` |
| `test::assert_eq_tensor` | `(&tensor[..r,p],&tensor[..r,p],string)->unit!{Test}` |
| `test::assert_shape` | `(&tensor[..r,p],List[int64],string)->unit!{Test}` |
| `time::add_days` | `(Date,int64)->Date` |
| `time::date` | `(int64,int64,int64)->Date` |
| `time::date_gt` | `(Date,Date)->bool` |
| `time::date_gte` | `(Date,Date)->bool` |
| `time::date_lt` | `(Date,Date)->bool` |
| `time::date_lte` | `(Date,Date)->bool` |
| `time::date_to_string` | `(Date)->string` |
| `time::day_of_week` | `(Date)->DayOfWeek` |
| `time::day_of_week_name` | `(Date)->string` |
| `time::day_of_year` | `(Date)->int64` |
| `time::days_between` | `(Date,Date)->int64` |
| `time::duration` | `(int64,int64,int64,int64)->Duration` |
| `time::is_leap_year` | `(int64)->bool` |
| `time::parse_date` | `(string)->Option[Date]` |
| `time::sub_days` | `(Date,int64)->Date` |
| `time::try_date` | `(int64,int64,int64)->Option[Date]` |
| `tokenizer::batch_encode` | `(Tokenizer,List[string],int64,int64)->tensor[batch,seq,int64]` |
| `tokenizer::decode` | `(Tokenizer,List[int64])->string` |
| `tokenizer::encode` | `(Tokenizer,string)->List[int64]` |
| `tokenizer::load_tokenizer` | `(string)->Tokenizer!{IO}` |
| `tokenizer::try_load_tokenizer` | `(string)->Option[Tokenizer]!{IO}` |""".splitlines()
    ),
    "05-OP-38": tuple(
        """\
> | `tensor_scan` | `(T,((T,int64)->T!E),int64)->tensor[n,T]!E` |
> | `process_run` | `(string,List[string])->(int64,string,string)!{IO}` |
> | `test_assert_eq` | `(Q,Q,string)->unit!{Test}` |
> | `test_assert_close_tensor` | `(&tensor[..r,p_float],&tensor[..r,p_float],p_float,string)->unit!{Test}` |
> | `test_assert_eq_tensor` | `(&tensor[..r,p],&tensor[..r,p],string)->unit!{Test}` |""".splitlines()
    ),
}

FROZEN_ATOM_DIGESTS = {
    "04-LIN-3": "52a61c21d53b8eaf194feebed4eee608f49bc30ebb0008fcc4d366fd93c3e649",
    "04-LIN-4": "ab21050a84236c40236b7d8d53453767dc15839a44ae1fd012d33bed411fecf9",
    "04-LIN-5": "2ad4e07442bf890a6fdd434362f50ab86215d6bd35c3d680e515de6fba5f9a29",
    "04-LIN-6": "6cfcc780a3f5b9836507772cf7ef76ce231a0505f07e9a06b3c1ba55e0946d92",
    "04-LIN-7": "6c1d8d77d251d245df6e1aa6e2138458048bdac31a407adfeba586302b0f3625",
    "04-LIN-8": "3e0013311e070716145da9245eea66361c8cf91fc6914ad200b68790b41313eb",
    "04-NUM-2": "1aab318622574c9505ec5e85472b27bf333318657407c38b2311325962e19a96",
    "04-NUM-4": "685b5a3447a069f138877d357e65d1ab225e6b712e62b2a5bd38e1ef960636cb",
    "04-NUM-8": "8887537f42a0c8263569296700826dc7466a0a3bf05e5c854ffff2406f075028",
    "04-NUM-11": "903b437e9aaa98b7c4d7c0c019393bee203d8bf76621902fc2ad53d872945f3f",
    "04-NUM-14": "621e87291569ed74f24adf9a9a1a2092b67a6824ef985ceb2645f6502c63f786",
    "04-NUM-16": "939c10f9449bb91c3117ec6d66f8afde5bedb733dec88be1623c7110740b8053",
    "04-SHAPE-1": "0f3f3f71731481b457b226bbcc8877d54962d268ec8787fbc6aa56ab4324d17c",
    "05-OP-1": "c2fb6c19db7ada4f86af7436f4f531ee0adb1395c7080ea94b25fe2e0f0b8d6c",
    "05-OP-2": "86fe2002cebd6192d15078ed0e8144936e38ba14f925802d2d526b7bf880ecd3",
    "05-OP-3": "b5a3ee9ca9a4f3161e20e729467d044878080ac8fb302af14b512bea66a58d3b",
    "05-OP-4": "f45693d5e3ef37033aef4c9a3f03de1806ea034d8247390c3ff6113d1c2ffa13",
    "05-OP-5": "00b9b1ecfdb42d0def6cc38296a518d9a25039cb2af3bdac53f1648225377d9a",
    "05-OP-6": "95d842566c76f85e0e89844029d7387921f57f6997e68107be92fe9b9cc1061c",
    "05-OP-7": "d3c5120918a8de774833d776204d62c01b3eccd01ffc22e43ff5422d4e28e54b",
    "05-OP-8": "ea385826c01b7cb1d24e75e1dbb4889149f0a08441d798eafcee75fc7d9f7b4f",
    "05-OP-9": "8359a6d8688f86f3818477c4d3fd8df04018e593fab9ad7ffa6fd0ebf5c1acf1",
    "05-OP-10": "5d77be3eb92d9db44da15ea02a0706239f8f1c70b32ab0391d5dad6aff94cfa2",
    "05-OP-11": "688952d434f31b332cd82a871a4da3e0a6e64e2d440f258ff9d24e5be8a37945",
    "05-OP-12": "6e99f2ecbb3c7fbf3ae4f852104b03bdf65f5edf7ead1cae02c9e1d833708353",
    "05-OP-13": "3fd84ffa594776abc51a8c277c5d9a0dbc2b7b8fc11ab1bf32209b30c7d97890",
    "05-OP-14": "cba4686a8a31fb18cf9c5213af5ac4a545b03ce548b5c7bcd96e8f7f76648a40",
    "05-OP-15": "a9ad8cf1423e99bf092e6dd52ce273555cda77b4beb52ff3a78564db1800933e",
    "05-OP-16": "b5ee19533f9f7241117f5958e092978446b49faacba312eec5781be548b3303d",
    "05-OP-17": "7c85fa51323c9993bb63f97ee4728c8fc5c13bd2c30315f6bcf7a69a77a39304",
    "05-OP-18": "6cf5eaa4e1ab068d667ea1d8cc26ba366694329669cba39443a4878a6d04714a",
    "05-OP-19": "d84cd2202079d852ba918b99e2bae1c964650362dee200942a9addba954bd5e5",
    "05-OP-20": "302cee9a558337a469751b4a5ec3ef009a2ee5ef5d9c68e32fd40c23cf1479f1",
    "05-OP-21": "ced775b654a61c4d36d2313191145b3543e55ef75825a5066e04b3c114437545",
    "05-OP-22": "f7c7c00b0fbea5176eb3427b517f5fb9f7434e24caaacd86fc1408455658329a",
    "05-OP-23": "1e0adc2fc7abf416c131f9ad5b6b054581eabf0365a9707b985fbaaa06e5e5b7",
    "05-OP-24": "2f3009f8b80b944f11fa2cf378409d85eb7891a59cdc1024e58ce2b67af5b800",
    "05-OP-25": "29f27a57545efd179e0f6b2766f48d7fdafb7c2ac6c3bbaeeaf5b0c0646ed502",
    "05-OP-26": "90050a454489c33ba0afb9caa41763591f22461eca947dd3525109c97976362f",
    "05-OP-27": "03a81560ae84cb4dd151e57da34d117a9e616a33700957796edea98c2afaf82f",
    "05-OP-28": "9eb81ed515be3e016371f951a75a3b65c4bae2cd8bfbc8de22c510f8e71be56b",
    "05-OP-29": "3fc46cb450b49244dfea8859a662190128420ab2565f7d18f5f97d7ffb27fd0a",
    "05-OP-30": "30c8c04f547161b7c40cbe5659a0c5fee34102f34a6fc605bcde8740221b461b",
    "05-OP-31": "31e1d9d5be4b12496c6a5f9d3ee3cd8f134d9868e2c0e526bb36dea97b813538",
    "05-OP-32": "e8102df69288ef68e023b236ef6e74bd82b327b50fd94f9aa9880cc6c8dfdeeb",
    "05-OP-33": "b0c9cf420f89f9c4c74219a41e55fd2bd36619bba25989a3d06a302424764fca",
    "05-OP-34": "0d2c7d4a051a43dc6b0c93b241434ff1d66bbd7a3e6d47e5c74b669d2fd687bf",
    "05-OP-35": "6eb9a0e1023aeed6dcf43abe8623a9b94dcb38db15224f38915320108c276ef7",
    "05-OP-36": "aeaaf9888f922b31159b8b7536444603897d649c8fb477e77bda659346177ab4",
    "05-OP-37": "2b27734c6956b706e031130b2c444cb69f0ff5a6a6886d1935611f61767f02b0",
    "05-OP-38": "46685ae74bc05fac13ce0ff877e92978bea109b4660f7d7ca637222819cb368e",
    "05-OP-39": "c23d7e9e0964df3655319ce26c486a8c006097cdb8ff714d8f7a1b8fcfecaa14",
    "05-OP-40": "4d3da3d13beca3c53b18515dbd15178b214fa3baa015ab975257ef778899dc18",
    "05-OP-41": "7bbbba7450bf89f9eac66a7f660f7352940a41e4baf6f7497873e46a29be41db",
    "05-OP-42": "d469e00652b7b9239f37532817b3c0f563bf22a66743c66dab66ce879be43ff4",
    "05-OP-43": "51dd3a7b7df5ecc20c7796a49f7a0122daf0f3a4b6538993964fea7a9f284ee7",
}

# The markers are part of the freeze contract: each must occur exactly once,
# and the end marker is excluded from the digest. Digests are not a self-bless
# mechanism. An intentional change owes the owning spec/design update, every
# consuming contract, and an adversarial mutation before this manifest moves.
FROZEN_REGION_DIGESTS = {
    "agent numeric surface discipline": (
        "AGENTS.md",
        "### Numeric Surface Discipline",
        "### Public-Surface Change Rule",
        "de1f56b43a92495cc8a71d7e543b888372f4fb803946a1fed78eb603fb67f917",
    ),
    "numeric value semantics": (
        "spec/04-type-system.md",
        "## 9. Numeric Value Semantics",
        "## 10. Checker Totality",
        "cd8beb14a6a9763ae9bd9409035e04a44726da35fd94185748ce8c042b215215",
    ),
    "numeric primitive contracts": (
        "spec/05-risc-primitives.md",
        "### 2.1 Elementwise Binary",
        "### 2.4 Movement",
        "90fdc968d6de13e06f0908da4666d98d6e68258195eed3123e30b2c57dda7485",
    ),
    "logical builtin contract": (
        "spec/05-risc-primitives.md",
        "### 3.2 Comparison and Logical Operations",
        "### 3.3 Activation Functions",
        "0f5112fa156d699d2bff79b65c0447d1075a381556b5f887f57d8dd0646dfd57",
    ),
    "name-preserving rank polymorphism": (
        "spec/04-type-system.md",
        "#### 4.5.3 Name-Preserving Rank Polymorphism",
        "#### 4.5.4 Concat Result Typing",
        "8f7ed49ec2a00115a9fc9d423c0e56040f68b42d31580b259597da98ec859733",
    ),
    "window extrema contract": (
        "spec/05-risc-primitives.md",
        "### 2.3.1 Windowed Reduction",
        "### 2.4 Movement",
        "cc2096d327307abec84098d50e4788d1a599f8ce2fadd930541ee078e1d7c786",
    ),
    "to_string contract section": (
        "spec/05-risc-primitives.md",
        "### 3.6.3 Canonical value-to-string conversion",
        "### 3.7 Host-Lane Data I/O Numeric Operations",
        "8520543d576332fce2cd9821a3d917b083bfeb54e16f2054ccc7b94217dcecda",
    ),
    "named lossy cast section": (
        "spec/05-risc-primitives.md",
        "### 3.8 Named Lossy Cast Forms",
        "## 4. Standard Lowerings",
        "4c2e336e26005cfea1a4cd649236231db3f5a4add35305f385434bbd3c68f089",
    ),
    "capability schema": (
        "spec/design/capability_table.md",
        "## The two-table design",
        "## Seed dispositions the table must ship with",
        "e623262cd4cd37b6646c651fd0596435cf28f24423b8d9d755620d009520c09d",
    ),
    "capability seed dispositions": (
        "spec/design/capability_table.md",
        "## Seed dispositions the table must ship with",
        "## New numeric ops before the table lands (added 2026-07-30)",
        "1f30b0e2a015b226a6c2eb5b0a18d27a9ae0c027cef5378ed963811a8b2fbe7f",
    ),
    "Phase 4 handoff": (
        "spec/design/dtype_semantics.md",
        "## Phase 4 - the capability table becomes the permanent guard",
        "## I1. Interlock with loud unsupported ([#730])",
        "1b3c5caa7e56a57770842270a166fb648265595c8a1ff42143c5b060713091f9",
    ),
    "compiled stdlib consumer": (
        "spec/design/loud_unsupported.md",
        "### LU5 - derived compiled-stdlib acceptance corpus ([#955])",
        "### LU6 - exhaustive checked host-cast planning ([#1150])",
        "30ba46096ab69975228a288c42633ace364da7c5fe92307fd696d8100df3a17e",
    ),
    "provenance governed surfaces": (
        "spec/design/spec_provenance.md",
        "| Capability-row, Deep-tag, tolerance-row, and diagnostic citations |",
        "| OpenSpec proving inadequate as the trigger for provenance work |",
        "56675ebf30876d376c9eeb1817472381d9cfb0254522f57a3894366762da782c",
    ),
    "roadmap ownership": (
        "spec/design/remediation_roadmap.md",
        "| **v0.19.0 - grounded dtype storage break",
        "| **v0.20.0 - behavior-preserving permanent guards**",
        "ea3f2741a9f9473d0d5e059b9d0c8bcb6454fe299734afe64ce2cfa5b9eb06eb",
    ),
    "status dtype row": (
        "docs/investigations/remediation_status_2026_08_04.md",
        "| **#729 dtype semantics** |",
        "| **#730 loud unsupported** |",
        "bbb8be3bf01b8143b368b59211a1f123318a53df2a8f6b803d088878c2c34793",
    ),
}


class OracleError(RuntimeError):
    """The frozen Phase 4B contract is incomplete or internally inconsistent."""


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
    "05-OP-31": "spec/registry/c_scalar_carrier.md",
    "05-OP-32": "spec/registry/c_container_boundary.md",
    "05-OP-33": "spec/registry/c_tensor_runtime.md",
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


def strict_atom_block(text: str, atom: str) -> str:
    starts = [match for match in ATOM_START.finditer(text) if match.group(1) == atom]
    if len(starts) != 1:
        raise OracleError(
            f"frozen normative atom {atom} must occur exactly once, got {len(starts)}"
        )
    start = starts[0].start()
    end = start
    for line in text[start:].splitlines(keepends=True):
        if end > start and ATOM_START.match(line):
            break
        if not line.startswith(">"):
            break
        end += len(line)
    return text[start:end]


def normalize_frozen_block(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) + "\n"


def frozen_digest(text: str) -> str:
    return hashlib.sha256(normalize_frozen_block(text).encode("utf-8")).hexdigest()


def frozen_region(text: str, start: str, end: str, label: str) -> str:
    start_count = text.count(start)
    end_count = text.count(end)
    if start_count != 1 or end_count != 1:
        raise OracleError(
            f"frozen {label} markers must occur exactly once "
            f"(start={start_count}, end={end_count})"
        )
    start_index = text.index(start)
    end_index = text.index(end)
    if end_index <= start_index:
        raise OracleError(f"frozen {label} end marker precedes its start")
    return text[start_index:end_index]


def validate_frozen_contract(
    docs: dict[str, str], violations: list[str]
) -> None:
    # A region digest cannot defend its own boundaries: contradictory prose can
    # otherwise be inserted immediately before its start or after its end. The
    # file snapshot is therefore the additive-contradiction gate. Narrower
    # atom and region digests remain below to identify the owning contract when
    # an existing clause changes. Moving a file digest is a semantic freeze
    # change and owes the same spec, consumer, and adversarial-test review as a
    # region-digest change.
    for relative, expected in FROZEN_FILE_DIGESTS.items():
        actual = frozen_digest(docs[relative])
        if actual != expected:
            violations.append(
                f"frozen contract file {relative} digest mismatch: "
                f"expected {expected}, got {actual}"
            )

    for atom, expected in FROZEN_ATOM_DIGESTS.items():
        relative = (
            "spec/04-type-system.md" if atom.startswith("04-") else "spec/05-risc-primitives.md"
        )
        try:
            actual = frozen_digest(strict_atom_block(docs[relative], atom))
        except OracleError as error:
            violations.append(str(error))
            continue
        if actual != expected:
            violations.append(
                f"frozen normative atom {atom} digest mismatch: "
                f"expected {expected}, got {actual}"
            )

    for label, (relative, start, end, expected) in FROZEN_REGION_DIGESTS.items():
        try:
            block = frozen_region(docs[relative], start, end, label)
        except OracleError as error:
            violations.append(str(error))
            continue
        actual = frozen_digest(block)
        if actual != expected:
            violations.append(
                f"frozen {label} digest mismatch: expected {expected}, got {actual}"
            )


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
    agents = docs["AGENTS.md"]
    spec02 = docs["spec/02-surf-syntax.md"]
    spec03 = docs["spec/03-deep-syntax.md"]
    spec04 = docs["spec/04-type-system.md"]
    spec05 = docs["spec/05-risc-primitives.md"]
    spec06 = docs["spec/06-transformations.md"]
    spec10 = docs["spec/10-serialization.md"]
    spec11 = docs["spec/11-ffi.md"]
    ownership_design = docs["spec/design/compiled_value_ownership.md"]
    implicit_linearity = docs["spec/design/implicit_linearity.md"]
    captured_risc = docs["openspec/specs/risc-primitives/spec.md"]
    captured_transformations = docs["openspec/specs/transformations/spec.md"]

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
        ),
        violations,
    )
    require_all(
        spec04,
        (
            ("The active primitive set is exactly ten names", "ten active primitives"),
            (
                "one of the ten active primitives is well-typed",
                "backend-neutral active primitive set",
            ),
            ("nine active tensor element dtypes", "nine tensor element dtypes"),
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
            ("Schema version 6 is explicitly\npresent", "wire v6 presence"),
            ("the only accepted version", "wire v6 exactness"),
            ("There is no versionless default", "wire versionless rejection"),
            ("versionless default, legacy migration", "wire migration rejection"),
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
                "No v5\nnumeric-fill migration or inferred fill dtype exists",
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
                "| `string` | `String` | opaque `chelis_string` handle and "
                "`CHELIS_VALUE_STRING` |",
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
                "`Unimplemented { issue: #879 }` before ownership verification "
                "constructs a\nplan",
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
                "The existing exact ABI remains [05-OP-31..33] and all three registries",
                "complete C ABI authority chain",
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
                "Runtime-valued `with seed` remains [#735] syntax/semantics work; "
                "recursive-host operation support remains [#729]/[#730] capability "
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
                "The current C/HIP emitters still\n"
                "treat the IR node as an emission no-op and reconstruct host "
                "releases from backend-local\nstate",
                "current Drop implementation status",
            ),
            (
                "the successor verified-ownership lanes emit the matching heap "
                "release at the\nterminal operation",
                "successor Drop release",
            ),
        ),
        violations,
    )
    require_all(
        agents,
        (
            (
                "No grandfather, permanent-disposition,",
                "agent zero-exception policy",
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
            "A language binding or device descriptor SHALL preserve rank as int32 "
            "and each extent, stride, element count, and byte capacity as int64",
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
            "Projection from the exact count into `int64`, `usize`, or a target "
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
                "`int32` | `int32`, `int64` | accumulator dtype `a`",
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
                "their defaults settle in source order",
                "positional expand settlement order",
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
            "ingest as `JsonInt` carrying its exact int64 value",
            "An integer-form token outside int64 range SHALL ingest as",
            "`JsonBigInt` carrying the token's exact decimal spelling",
            "never\n> selects a lossy float image for an integer-form token",
            "CSV cells are TEXT at parse time",
            "integer accessors accept only its integer subset",
            "An empty or non-conforming cell is a loud error",
        ),
        "05-OP-3": (
            "`io/json::json_int` returns the stored `JsonInt` int64 exactly",
            "`io/json::json_float` returns a stored `JsonFloat` f64 exactly",
            "It never truncates or rounds a float into an integer",
            "`csv_int` | `(List[Dict[string,string]], int64, string) -> int64`",
            "`csv_ints` | `(List[Dict[string,string]], string) -> List[int64]`",
            "`csv_f64` | `(List[Dict[string,string]], int64, string) -> f64`",
            "`csv_f64s` | `(List[Dict[string,string]], string) -> List[f64]`",
            "`csv_nrows` | `(List[Dict[string,string]]) -> int64`",
            "no JSON variant or default cell is fabricated",
            "structurally rejected inside `grad`",
            "They have no accumulator",
        ),
        "05-OP-4": (
            "`JsonFloat(value)` accepts exactly f64",
            "`JsonInt(value)` accepts exactly int64",
            "every other operand width is a type error",
            "No construction path widens or narrows a numeric value",
            "feeds the byte-exact serialization channel of [05-OP-5]",
        ),
        "05-OP-5": (
            "emits a stored `JsonInt` int64 as its exact decimal digits",
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
            "returns the stored extent of `x` along `axis` as an exact `int64`",
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
            "complete before the operation consumes a Random call ordinal",
            "failure traps `Domain` as `uniform_like` and consumes none",
            "Equal bounds are valid and produce that stored value",
            "For `p = f64`, the element is the one f64 fused multiply-add",
            "For `p = f32`, it is the one f32 fused multiply-add",
            "For `p = f16` or `bf16`, the stored bounds widen exactly to f32",
            "their difference and the fused multiply-add execute once in f32",
            "the result narrows exactly once to `p`",
            "There is no f32 public-bound signature, default bound, or f64 "
            "intermediate",
            "introduces `Random`",
            "does not observe the template's element values",
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
            "`expand(g / divisor, original_shape, axis)`",
            "no accumulator parameter of its own",
        ),
        "05-OP-12": (
            "every active signed\n> integer and float tensor dtype",
            "compared without conversion at their stored dtype",
            "first NaN in increasing axis-index order",
            "first stored representation among equal\n> values",
            "execution-time extent is zero",
            "equal positive or negative infinities",
            "tie count `k` is\n> counted exactly as `int64`",
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
            "returns `int64` indices",
            "lowest\n> axis index containing NaN",
            "Comparisons never convert through another dtype",
            "execution-time extent is zero",
            "result dtype `int64`",
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
            "a tensor whose element dtype is one of the nine active tensor element dtypes",
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
            "returns an `int64` tensor",
            "one or more unique named axes",
            "Missing axes, mixed positional/named axes, duplicate normalized "
            "positions or names",
            "visited in original row-major order",
            "checked `int64` addition",
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
            "typedef struct { void *data; const int64_t *shape; const int64_t "
            "*strides; int64_t size; int64_t byte_capacity; int32_t rank; "
            "chelis_dtype dtype; uint8_t owns_data; uint8_t reserved[2]; } "
            "chelis_tensor;",
            "typedef struct { chelis_value key; chelis_value value; } "
            "chelis_dict_entry;",
            "rank in `0..=INT32_MAX`",
            "rank zero has null `shape` and `strides` pointers",
            "positive rank has non-null pointers to exactly `rank` int64 entries",
            "There is no rank-eight limit",
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
            "All lengths, indices, offsets, sizes, and returned counts are exact `int64`",
            "Negative lengths, indices, offsets, and counts trap `Domain`",
            "result-length and allocation arithmetic traps `Overflow`",
            "half-open increasing sequence",
            "Unicode scalar values",
            "A slice whose nonnegative start is at or beyond the scalar length "
            "is empty",
            "Dictionary keys are exactly `string`, `bool`, or a scalar of any active "
            "signed-integer dtype",
            "Equality includes the key kind and integer dtype",
            "Float keys are rejected",
            "later duplicate replaces the value",
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
            "exactly the twenty-three final public C callable identities",
            "axes and rank are `int32_t`",
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
            "no int32/int64-only dispatch exception",
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
            "`process::run` | `(string,List[string])->(int64,string,string)!{IO}`",
            "`contracts::normal_cdf` | `(p_float)->p_float`",
            "`init/random::normal_like` | "
            "`(&tensor[..r,p_float],p_float,p_float)->tensor[..r,p_float]!{Random}`",
            "`tensor/construct::linspace` | "
            "`(p_float,p_float,int64)->tensor[n,p_float]`",
            "`tensor/construct::arange` | "
            "`(p_int,p_int)->tensor[n,p_int]`",
            "`sort::sort` | `(&tensor[..r,p_numeric],int32)->"
            "(tensor[..r,p_numeric],tensor[..r,int64])`",
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
            "`(List[tensor[..pre,..post,p]],int32)->tensor[..pre,rows,..post,p]`",
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
            "requires finite endpoints and int64 `count >= 1`",
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
            "int64 extents",
            "compares its length and every entry to the tensor's complete shape "
            "in axis order",
            "Each finite element pair computes `abs(actual - expected)` at that "
            "width and is close exactly when that difference is less than or equal "
            "to the converted tolerance",
            "without invoking a shell",
            "every random stdlib callable has the pathwise adjoint of its exact "
            "graph above",
            "source units and mask comparisons contribute zero cotangent",
            "rounded result equals the stored upper endpoint",
            "computed denominator must be finite and strictly positive",
            "`days_between(lhs,rhs) = ordinal(rhs) - ordinal(lhs)`",
            "final normalized `days` field has no int64 representation",
            "A negative year uses `-` followed by exactly "
            "`max(4, digits(|year|))` decimal digits",
            "`|year|` is the exact mathematical magnitude rather than an int64 `abs`",
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
            "validation completes before Random consumption",
            "failure traps `Domain` as `dropout` while consuming no call ordinal",
            "accepted call consumes exactly one ordinal",
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
            "Dropped elements contribute positive zero",
            "contributions combine by the canonical adjacent-pair balanced tree",
            "mask comparison itself has zero cotangent",
            "has no accumulator parameter",
        ),
        "05-OP-38": (
            "governs exactly these five numeric-capacity identities and signatures",
            "`tensor_scan` | `(T,((T,int64)->T!E),int64)->tensor[n,T]!E`",
            "`process_run` | `(string,List[string])->(int64,string,string)!{IO}`",
            "`test_assert_eq` | `(Q,Q,string)->unit!{Test}`",
            "`test_assert_close_tensor` | `(&tensor[..r,p_float],"
            "&tensor[..r,p_float],p_float,string)->unit!{Test}`",
            "`test_assert_eq_tensor` | `(&tensor[..r,p],&tensor[..r,p],string)"
            "->unit!{Test}`",
            "`T` is one active numeric or bool scalar type",
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
        "05-RNG-1": (
            "Every conforming evaluation of a `with seed(N)` program produces "
            "byte-identical random results",
            "compiler version and target do not vary this result",
            "high 53 bits divided by `2^53`",
            "Each entered random primitive consumes exactly one call ordinal",
            "validation that precedes Random consumption consumes none",
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
            "no int32/int64-only exception",
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

    require_all(
        captured_risc,
        (
            (
                "`sub`, `max_elem`, and `min_elem` are Tier-1\nidentities",
                "captured direct arithmetic identities",
            ),
            (
                "it remains direct checked subtraction rather than becoming "
                "`add(a, neg(b))`",
                "captured direct sub lowering",
            ),
            (
                "it remains direct selection rather than becoming "
                "`neg(max_elem(neg(a), neg(b)))`",
                "captured direct min_elem lowering",
            ),
            (
                "first NaN in operand order with exact stored bits",
                "captured extrema stored-bit selection",
            ),
        ),
        violations,
    )
    require_all(
        captured_transformations,
        (
            (
                "the selected operand receives the whole cotangent,\n  including "
                "the first operand on equality",
                "captured extrema tie rule",
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
                "`Random | Accum | IO | Test | Resource(ResourceId)`",
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
                "Unimplemented { issue: #1281, diagnostic_kind: UnsupportedFeature }",
                "reduction implementation owner",
            ),
            (
                "Unimplemented { issue: #1290, diagnostic_kind: UnsupportedFeature }",
                "product implementation owner",
            ),
            (
                "Unimplemented { issue: #1284, diagnostic_kind: UnsupportedFeature }",
                "logical implementation owner",
            ),
            (
                "arithmetic reductions x (any) x bool",
                "bool reduction rejection",
            ),
            ("[05-OP-29], [#1287], [#1291]", "count capability owner"),
            (
                "`max_elem`/`min_elem` x Scalar/Tensor x active numeric dtypes "
                "([05-OP-40], [#715], [#1306])",
                "extrema capability owner",
            ),
            (
                "`sub` x Scalar/Tensor x active numeric dtypes "
                "([05-OP-41], [#1306])",
                "sub capability owner",
            ),
            (
                "Unimplemented { issue: #1306, diagnostic_kind: "
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
                ".venv/bin/python scripts/dtype_count_oracle.py",
                "Count child oracle command",
            ),
            ("DTYPE COUNT ORACLE: PASS", "Count child oracle success line"),
            (
                "checker grammar, dedicated non-alias\n`Count` IR",
                "Count child oracle ownership",
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
    validate_frozen_contract(docs, violations)
    if violations:
        raise OracleError("; ".join(violations))


def run_oracle(python: str = sys.executable, root: Path = REPO_ROOT) -> None:
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


if __name__ == "__main__":
    run_oracle()
