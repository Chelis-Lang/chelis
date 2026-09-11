//! The capacity census tripwire: chelis#729 / `spec/design/dtype_semantics.md`
//! §C6 deliverable 1 (PR #950), the pre-Phase-1 guard of the surface-growth
//! ratchet.
//!
//! It enumerates the public numeric surface from the ARTIFACTS (never a
//! hand-maintained list): the published runtime headers' transitive
//! `#include "..."` closure, and the desugared Deep AST of every
//! `packages/chelis-std/src` source. The inventory is diffed against the
//! checked-in baseline `spec/design/capacity_census.json`.
//!
//! Sanctioned actions when this test fails (also printed in the failure
//! message, which is the contract - a context-poor agent reads only that):
//!
//! 1. Every new or changed descriptor must land in exactly one final class:
//!    an exact structurally nonnumeric registration, an exact structurally
//!    recognized tagged transport, or an exact numeric-operation registration
//!    against its governing normative `[05-OP-N]` atom.
//! 2. The primary census has no transition authority. Citations,
//!    grandfathers, permanent dispositions, successor overrides, and
//!    maintainer overrides cannot authorize any identity.
//! 3. A removed row is an ABI removal and is 0.19 payload by default
//!    (`spec/design/remediation_roadmap.md` anti-churn invariant 7).
//!
//! This file and the baseline are guard artifacts: editing either to make a
//! change pass is never the fix. Wire-schema numeric fields and binding-side
//! raw-dtype parameters have typed, fixed executable manifest entries.
//! The wire aggregate requires final authority and execution receipts;
//! relabeling JSON cannot claim an enumerator or mutation oracle exists.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use chelis_deep::tag::DeepTag;
use chelis_deep::{Atom, Expr, List};
use chelis_types::types::Prim;
use serde::{Deserialize, Serialize};

#[path = "../../../tests/support/capacity_census_stdlib_tests.rs"]
mod stdlib_closure_tests;

#[path = "../../../tests/support/c_lexical.rs"]
mod c_lexical;

#[path = "../../../tests/support/c_preprocessor.rs"]
mod c_preprocessor;

#[path = "../../../tests/support/c_include.rs"]
mod c_include;

#[path = "../../../tests/support/capacity_census_backend_headers.rs"]
mod backend_headers;

#[path = "../../../tests/support/capacity_census_authority.rs"]
mod capacity_census_authority;
use capacity_census_authority::{
    AuthorityRegistries, NumericOperationRegistration as FinalNumericOperationRegistration,
    StaticSurfaceDescriptor, SurfaceDescriptor,
};

const BASELINE_REL: &str = "spec/design/capacity_census.json";
const INCLUDE_DIR_REL: &str = "crates/chelis-runtime/include";
/// Roots of the published header surface: `chelis_runtime.h`'s transitive
/// closure is the tarball surface; `chelis_blas.h` and `chelis_math.h` are
/// emitted as `#include`s by the C backend, so generated programs compile
/// against them too.
const HEADER_ROOTS: &[&str] = &["chelis_runtime.h", "chelis_blas.h", "chelis_math.h"];
const STD_SRC_REL: &str = "packages/chelis-std/src";
const CONTROLLING_SPEC_REL: &str = "spec/05-risc-primitives.md";
static PLANTED_INCLUDE_SEQUENCE: AtomicU64 = AtomicU64::new(0);
const NUMERIC_PRIMS: &[&str] = &[
    "f64", "f32", "f16", "bf16", "f8e4m3", "int8", "int16", "int32", "int64",
];
/// The subset of `NUMERIC_PRIMS` whose appearance in an untagged public
/// position is a capacity SEAM, mirroring `double`/`float` on the C side.
const FLOAT_PRIMS: &[&str] = &["f64", "f32", "f16", "bf16", "f8e4m3"];

/// What the census must do with one language primitive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrimCensusClass {
    /// A capacity SEAM in an untagged public position (no citation path).
    Float,
    /// A `numeric-op`; a new post-ratchet row with this class owes registration.
    Integer,
    /// Carries no dtype, so it is not census surface at all.
    NonNumeric,
}

/// Census membership of one `chelis_types::Prim`, decided by an EXHAUSTIVE
/// match. `NUMERIC_PRIMS`/`FLOAT_PRIMS` are string lists because the
/// desugared AST carries `(t-prim {} <name>)` SYMBOLS, not enum values, so
/// nothing about widening `Prim` would otherwise reach this file: a new
/// dtype could enter the language and every stdlib signature carrying it
/// would enumerate as dtype-free. The match is the lock. Adding a variant
/// makes this file stop compiling until the dtype is classified; removing
/// one makes `ALL_PRIMS` below stop compiling; and
/// `census_prim_lists_cover_every_prim_variant` checks this verdict against
/// the two string lists in both directions. `chelis-types` uses the same
/// device one layer down for its own active-set rules.
fn prim_census_class(prim: Prim) -> PrimCensusClass {
    match prim {
        // A dtype RESERVED but inactive (`spec/04-type-system.md` §1.1.1) is
        // still float for census purposes. The census asks what a spelling
        // would carry across a public boundary, not whether the checker
        // admits it today; letting those two answers diverge is how a
        // reserved dtype becomes a silent seam on the day it activates.
        Prim::F32 | Prim::F64 | Prim::F16 | Prim::Bf16 | Prim::F8e4m3 => PrimCensusClass::Float,
        Prim::Int8 | Prim::Int16 | Prim::Int32 | Prim::Int64 => PrimCensusClass::Integer,
        Prim::Bool | Prim::String => PrimCensusClass::NonNumeric,
    }
}

/// Closed enumeration of `chelis_types::Prim`, paired with the exhaustive
/// match above: a variant added to the enum breaks the match, a variant
/// removed from it breaks this list.
const ALL_PRIMS: &[Prim] = &[
    Prim::F32,
    Prim::F64,
    Prim::F16,
    Prim::Bf16,
    Prim::F8e4m3,
    Prim::Int8,
    Prim::Int16,
    Prim::Int32,
    Prim::Int64,
    Prim::Bool,
    Prim::String,
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct Row {
    kind: String,
    id: String,
    #[serde(default)]
    flags: Vec<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    citation: String,
}

const PRIMARY_CENSUS_FAMILY: &str = "covered-family";

/// Exact structural proofs for rows which carry no numeric capacity. Empty
/// derived flags are necessary but never sufficient: the complete family,
/// kind, canonical identity, and flag vector must match one of these rows.
const FINAL_NONNUMERIC_ROWS: &[StaticSurfaceDescriptor] = &[
    // A closed diagnostic identity; no extent, width, dtype, or payload is encoded.
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-struct",
        "chelis_runtime.h: typedef enum { CHELIS_REDUCE_SUM = 0 , CHELIS_REDUCE_COUNT = 1 , CHELIS_REDUCE_MAX = 2 , CHELIS_REDUCE_MIN = 3 , CHELIS_REDUCE_PROD = 4 , CHELIS_REDUCE_ARGMAX = 5 , CHELIS_REDUCE_ARGMIN = 6 } chelis_reduction_op",
        &[],
    ),
    // Closed sparse-operation identities select shape policy and diagnostics only.
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-struct",
        "chelis_runtime.h: typedef enum { CHELIS_SPARSE_GATHER = 0 , CHELIS_SPARSE_ADD = 1 , CHELIS_SPARSE_REPLACE = 2 , CHELIS_SPARSE_ELEMENTS = 3 } chelis_sparse_op",
        &[],
    ),
    // Closed movement operation/side identities contain no numeric payload.
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-struct",
        "chelis_runtime.h: typedef enum { CHELIS_MOVEMENT_EXPAND = 0 , CHELIS_MOVEMENT_INSERT = 1 , CHELIS_MOVEMENT_PAD = 2 , CHELIS_MOVEMENT_SHRINK = 3 , CHELIS_MOVEMENT_STRIDE = 4 } chelis_movement_op",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-struct",
        "chelis_runtime.h: typedef enum { CHELIS_MOVEMENT_SOURCE = 0 , CHELIS_MOVEMENT_RESULT = 1 } chelis_movement_side",
        &[],
    ),
    // Closed window operation/side identities select a contract, not a number.
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-struct",
        "chelis_runtime.h: typedef enum { CHELIS_WINDOW_SUM = 0 , CHELIS_WINDOW_MEAN = 1 , CHELIS_WINDOW_MAX = 2 , CHELIS_WINDOW_MIN = 3 , CHELIS_WINDOW_GRAD = 4 } chelis_window_op",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-struct",
        "chelis_runtime.h: typedef enum { CHELIS_WINDOW_SOURCE = 0 , CHELIS_WINDOW_RESULT = 1 } chelis_window_side",
        &[],
    ),
    // Closed matrix-part and dimension identities contain no numeric values.
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-struct",
        "chelis_runtime.h: typedef enum { CHELIS_MATMUL_LEFT = 0 , CHELIS_MATMUL_RIGHT = 1 , CHELIS_MATMUL_RESULT = 2 } chelis_matmul_part",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-struct",
        "chelis_runtime.h: typedef enum { CHELIS_MATMUL_ROWS = 0 , CHELIS_MATMUL_COLUMNS = 1 , CHELIS_MATMUL_REDUCTION = 2 } chelis_matmul_dimension_kind",
        &[],
    ),
    // Final nonnumeric lifetime operations from the retired primary cohort.
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: void chelis_adt_release ( const chelis_adt * adt ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: void chelis_adt_retain ( const chelis_adt * adt ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: void chelis_dict_release ( const chelis_dict * dict ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: void chelis_dict_retain ( const chelis_dict * dict ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: void chelis_list_release ( const chelis_list * list ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: void chelis_list_retain ( const chelis_list * list ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: void chelis_string_release ( chelis_string value ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: void chelis_string_retain ( chelis_string value ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: void chelis_tuple_release ( const chelis_tuple * tuple ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: void chelis_tuple_retain ( const chelis_tuple * tuple ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: void chelis_value_release ( chelis_value value ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: _Bool chelis_adt_tag_equals ( const chelis_adt * adt , chelis_string ctor ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: _Bool chelis_file_exists ( chelis_string path ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: _Bool chelis_string_contains ( chelis_string haystack , chelis_string needle ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: _Bool chelis_string_ends_with ( chelis_string value , chelis_string suffix ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: _Bool chelis_string_eq ( chelis_string lhs , chelis_string rhs ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: _Bool chelis_string_starts_with ( chelis_string value , chelis_string prefix ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: _Noreturn void chelis_fail ( chelis_string message ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: chelis_list * chelis_list_dir ( chelis_string path ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: chelis_list * chelis_read_lines ( chelis_string path ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: chelis_mapped_file * chelis_mmap_file ( chelis_string path ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: chelis_string chelis_adt_get_tag ( const chelis_adt * adt ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: chelis_string chelis_read_file ( chelis_string path ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: chelis_string chelis_string_concat ( chelis_string lhs , chelis_string rhs ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: chelis_string chelis_string_from_cstr ( const char * value ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: chelis_string chelis_string_trim ( chelis_string value ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: const char * chelis_string_data ( chelis_string value ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: void chelis_write_file ( chelis_string path , chelis_string contents ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: chelis_adt * chelis_adt_take_value ( chelis_value value ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: const chelis_adt * chelis_adt_borrow_value ( chelis_value value ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: chelis_dict * chelis_dict_take_value ( chelis_value value ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: const chelis_dict * chelis_dict_borrow_value ( chelis_value value ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: chelis_list * chelis_list_take_value ( chelis_value value ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: const chelis_list * chelis_list_borrow_value ( chelis_value value ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: chelis_mapped_file * chelis_mapped_file_take_value ( chelis_value value ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: const chelis_mapped_file * chelis_mapped_file_borrow_value ( chelis_value value ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: void chelis_mapped_file_retain ( const chelis_mapped_file * mapped ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: void chelis_mapped_file_release ( const chelis_mapped_file * mapped ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: chelis_option * chelis_option_none ( void ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: chelis_option * chelis_option_some ( chelis_value value ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: _Bool chelis_option_is_some ( const chelis_option * option ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: chelis_value chelis_option_unwrap ( const chelis_option * option ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: chelis_option * chelis_option_take_value ( chelis_value value ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: const chelis_option * chelis_option_borrow_value ( chelis_value value ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: void chelis_option_retain ( const chelis_option * option ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: void chelis_option_release ( const chelis_option * option ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: chelis_string chelis_string_take_value ( chelis_value value ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: chelis_string chelis_string_borrow_value ( chelis_value value ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: chelis_tensor_write * chelis_tensor_begin_write ( chelis_tensor * tensor ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: void chelis_tensor_end_write ( chelis_tensor_write * guard ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: void chelis_tensor_retain ( const chelis_tensor * tensor ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: void chelis_tensor_release ( const chelis_tensor * tensor ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: chelis_tensor * chelis_tensor_take_value ( chelis_value value ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: const chelis_tensor * chelis_tensor_borrow_value ( chelis_value value ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: chelis_tuple * chelis_tuple_take_value ( chelis_value value ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: const chelis_tuple * chelis_tuple_borrow_value ( chelis_value value ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: chelis_value chelis_value_clone ( chelis_value value ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: chelis_value chelis_value_take_adt ( chelis_adt * adt ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: chelis_value chelis_value_take_dict ( chelis_dict * dict ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: chelis_value chelis_value_take_list ( chelis_list * list ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: chelis_value chelis_value_take_mapped_file ( chelis_mapped_file * mapped ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: chelis_value chelis_value_take_option ( chelis_option * option ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: chelis_value chelis_value_take_string ( chelis_string value ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: chelis_value chelis_value_take_tensor ( chelis_tensor * tensor ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: chelis_value chelis_value_take_tuple ( chelis_tuple * tuple ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-struct",
        "chelis_runtime.h: typedef struct { void * handle ; } chelis_string",
        &[],
    ),
];

/// The exact tagged-carrier declarations of the chelis#1289 public C ABI:
/// the dtype tag enumeration, the canonical `chelis_scalar` image, the
/// tagged `chelis_value` (tag enum, payload union, carrier struct), and the
/// fixed tensor read/write views. Opaque heap descriptors are not transport
/// rows: only the view returned under [05-OP-44]'s access rules carries
/// numeric data across the ABI. These rows ARE the tagged transport; the
/// callables operating on them register as numeric operations below.
const FINAL_TAGGED_TRANSPORT_ROWS: &[StaticSurfaceDescriptor] = &[
    // Exact C6 container transports and [05-OP-31] dictionary-entry shape.
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: chelis_list * chelis_dict_entries ( const chelis_dict * dict ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: chelis_list * chelis_dict_keys ( const chelis_dict * dict ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: chelis_list * chelis_dict_values ( const chelis_dict * dict ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: chelis_list * chelis_list_append ( const chelis_list * list , chelis_value value ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: chelis_list * chelis_list_concat ( const chelis_list * lhs , const chelis_list * rhs ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: chelis_list * chelis_list_empty ( void ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: chelis_list * chelis_list_flatten ( const chelis_list * list ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-export",
        "chelis_runtime.h: chelis_list * chelis_list_zip ( const chelis_list * lhs , const chelis_list * rhs ) ;",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-struct",
        "chelis_runtime.h: typedef struct { chelis_value key ; chelis_value value ; } chelis_dict_entry",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-struct",
        "chelis_runtime.h: enum { CHELIS_VALUE_UNIT = 0 , CHELIS_VALUE_SCALAR = 1 , CHELIS_VALUE_STRING = 2 , CHELIS_VALUE_TENSOR = 3 , CHELIS_VALUE_LIST = 4 , CHELIS_VALUE_TUPLE = 5 , CHELIS_VALUE_DICT = 6 , CHELIS_VALUE_ADT = 7 , CHELIS_VALUE_OPTION = 8 , CHELIS_VALUE_MAPPED_FILE = 9 }",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-struct",
        "chelis_runtime.h: typedef struct { chelis_dtype dtype ; uint8_t reserved [ 7 ] ; uint64_t bits ; } chelis_scalar",
        &["numeric-op"],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-struct",
        "chelis_runtime.h: typedef struct { chelis_value_tag tag ; uint8_t reserved [ 7 ] ; chelis_value_payload payload ; } chelis_value",
        &["numeric-op"],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-struct",
        "chelis_runtime_views.h: typedef struct { const void * data ; int64_t count ; chelis_dtype dtype ; uint8_t reserved [ 7 ] ; } chelis_read_view",
        &["numeric-op"],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-struct",
        "chelis_runtime_views.h: typedef struct { void * data ; int64_t count ; chelis_dtype dtype ; uint8_t reserved [ 7 ] ; } chelis_write_view",
        &["numeric-op"],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-struct",
        "chelis_runtime.h: typedef union { chelis_scalar scalar ; void * handle ; } chelis_value_payload",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        PRIMARY_CENSUS_FAMILY,
        "header-struct",
        "chelis_runtime_dtype.h: enum { CHELIS_DTYPE_F32 = 0 , CHELIS_DTYPE_F64 = 1 , CHELIS_DTYPE_I32 = 2 , CHELIS_DTYPE_BOOL = 3 , CHELIS_DTYPE_I64 = 4 , CHELIS_DTYPE_BF16 = 5 , CHELIS_DTYPE_F16 = 6 , CHELIS_DTYPE_I8 = 7 , CHELIS_DTYPE_I16 = 8 }",
        &[],
    ),
];

/// Exact semantic registrations for every numeric public C callable of the
/// chelis#1289 tagged-carrier ABI. Each anchor is a literal phrase from the
/// governing atom's normative block in spec/05-risc-primitives.md; the atoms
/// incorporate their identity registries under spec/registry/ by reference.
macro_rules! final_numeric_row {
    ($kind:literal, $id:literal, $flags:expr, $atom:literal, $anchor:literal) => {
        FinalNumericOperationRegistration {
            surface: StaticSurfaceDescriptor::new(PRIMARY_CENSUS_FAMILY, $kind, $id, $flags),
            atom: $atom,
            authority_anchor: $anchor,
        }
    };
}

const FINAL_NUMERIC_OPERATION_ROWS: &[FinalNumericOperationRegistration] = &[
    final_numeric_row!(
        "header-export",
        "chelis_runtime.h: int64_t chelis_metadata_plan_byte_offset ( const chelis_metadata_plan * plan , chelis_scalar linear_index ) ;",
        &["numeric-op"],
        "[05-OP-33]",
        "`chelis_metadata_plan_byte_offset` takes a canonical tagged int64 logical"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_runtime.h: chelis_metadata_plan * chelis_metadata_plan_new ( chelis_scalar rank , const chelis_scalar * shape , chelis_scalar exemplar ) ;",
        &[],
        "[05-OP-33]",
        "`chelis_metadata_plan_new` constructs an independently owned, opaque"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_runtime.h: chelis_metadata_plan * chelis_metadata_plan_view ( chelis_scalar rank , const chelis_scalar * shape , const chelis_scalar * strides , chelis_scalar exemplar , chelis_scalar byte_capacity ) ;",
        &[],
        "[05-OP-33]",
        "`chelis_metadata_plan_view` instead retains rank-many exact tagged int64"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_runtime.h: int32_t chelis_metadata_plan_rank ( const chelis_metadata_plan * plan ) ;",
        &["numeric-op"],
        "[05-OP-33]",
        "`chelis_metadata_plan_new` constructs an independently owned, opaque"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_runtime.h: const int64_t * chelis_metadata_plan_shape ( const chelis_metadata_plan * plan ) ;",
        &["numeric-op"],
        "[05-OP-33]",
        "`chelis_metadata_plan_new` constructs an independently owned, opaque"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_runtime.h: const int64_t * chelis_metadata_plan_strides ( const chelis_metadata_plan * plan ) ;",
        &["numeric-op"],
        "[05-OP-33]",
        "`chelis_metadata_plan_new` constructs an independently owned, opaque"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_runtime.h: int64_t chelis_metadata_plan_count ( const chelis_metadata_plan * plan ) ;",
        &["numeric-op"],
        "[05-OP-33]",
        "`chelis_metadata_plan_new` constructs an independently owned, opaque"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_runtime.h: int64_t chelis_metadata_plan_byte_count ( const chelis_metadata_plan * plan ) ;",
        &["numeric-op"],
        "[05-OP-33]",
        "`chelis_metadata_plan_new` constructs an independently owned, opaque"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_runtime.h: chelis_dtype chelis_metadata_plan_dtype ( const chelis_metadata_plan * plan ) ;",
        &["numeric-op"],
        "[05-OP-33]",
        "`chelis_metadata_plan_new` constructs an independently owned, opaque"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_runtime.h: void chelis_metadata_plan_check_capacity ( const chelis_metadata_plan * plan , chelis_scalar byte_capacity ) ;",
        &[],
        "[05-OP-33]",
        "`chelis_metadata_plan_check_capacity` validates a canonical nonnegative int64"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_runtime.h: void chelis_metadata_plan_release ( chelis_metadata_plan * plan ) ;",
        &[],
        "[05-OP-33]",
        "`chelis_metadata_plan_release` consumes the live plan exactly once; all"
    ),
    // Final [05-OP-32] registrations from the retired primary cohort.
    final_numeric_row!(
        "header-export",
        "chelis_runtime.h: _Bool chelis_dict_contains ( const chelis_dict * dict , chelis_value key ) ;",
        &[],
        "[05-OP-32]",
        "shape_index(container, parameters...) -> result"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_runtime.h: chelis_adt * chelis_adt_construct ( chelis_string ctor , const chelis_value * fields , int64_t len ) ;",
        &["numeric-op"],
        "[05-OP-32]",
        "shape_index(container, parameters...) -> result"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_runtime.h: chelis_dict * chelis_dict_from_pairs ( const chelis_list * pairs ) ;",
        &[],
        "[05-OP-32]",
        "shape_index(container, parameters...) -> result"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_runtime.h: chelis_dict * chelis_dict_insert ( const chelis_dict * dict , chelis_value key , chelis_value value ) ;",
        &[],
        "[05-OP-32]",
        "shape_index(container, parameters...) -> result"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_runtime.h: chelis_dict * chelis_dict_remove ( const chelis_dict * dict , chelis_value key ) ;",
        &[],
        "[05-OP-32]",
        "shape_index(container, parameters...) -> result"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_runtime.h: chelis_list * chelis_list_chunk ( const chelis_list * list , int64_t size ) ;",
        &["numeric-op"],
        "[05-OP-32]",
        "shape_index(container, parameters...) -> result"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_runtime.h: chelis_list * chelis_list_drop ( const chelis_list * list , int64_t count ) ;",
        &["numeric-op"],
        "[05-OP-32]",
        "shape_index(container, parameters...) -> result"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_runtime.h: chelis_list * chelis_list_enumerate ( const chelis_list * list ) ;",
        &[],
        "[05-OP-32]",
        "shape_index(container, parameters...) -> result"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_runtime.h: chelis_list * chelis_list_from_values ( const chelis_value * items , int64_t len ) ;",
        &["numeric-op"],
        "[05-OP-32]",
        "shape_index(container, parameters...) -> result"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_runtime.h: chelis_list * chelis_list_take ( const chelis_list * list , int64_t count ) ;",
        &["numeric-op"],
        "[05-OP-32]",
        "shape_index(container, parameters...) -> result"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_runtime.h: chelis_list * chelis_mmap_read ( const chelis_mapped_file * mapped , int64_t offset , int64_t len ) ;",
        &["numeric-op"],
        "[05-OP-32]",
        "shape_index(container, parameters...) -> result"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_runtime.h: chelis_list * chelis_range_i64 ( int64_t start , int64_t end ) ;",
        &["numeric-op"],
        "[05-OP-32]",
        "shape_index(container, parameters...) -> result"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_runtime.h: chelis_list * chelis_read_bytes ( chelis_string path ) ;",
        &[],
        "[05-OP-32]",
        "shape_index(container, parameters...) -> result"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_runtime.h: chelis_string chelis_string_slice ( chelis_string value , int64_t start , int64_t len ) ;",
        &["numeric-op"],
        "[05-OP-32]",
        "shape_index(container, parameters...) -> result"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_runtime.h: chelis_tuple * chelis_tuple_from_values ( const chelis_value * items , int64_t len ) ;",
        &["numeric-op"],
        "[05-OP-32]",
        "shape_index(container, parameters...) -> result"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_runtime.h: chelis_value chelis_adt_get_field ( const chelis_adt * adt , int64_t index ) ;",
        &["numeric-op"],
        "[05-OP-32]",
        "shape_index(container, parameters...) -> result"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_runtime.h: chelis_value chelis_list_index ( const chelis_list * list , int64_t index ) ;",
        &["numeric-op"],
        "[05-OP-32]",
        "shape_index(container, parameters...) -> result"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_runtime.h: chelis_value chelis_tuple_get ( const chelis_tuple * tuple , int64_t index ) ;",
        &["numeric-op"],
        "[05-OP-32]",
        "shape_index(container, parameters...) -> result"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_runtime.h: int64_t chelis_adt_field_count ( const chelis_adt * adt ) ;",
        &["numeric-op"],
        "[05-OP-32]",
        "shape_index(container, parameters...) -> result"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_runtime.h: int64_t chelis_dict_len ( const chelis_dict * dict ) ;",
        &["numeric-op"],
        "[05-OP-32]",
        "shape_index(container, parameters...) -> result"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_runtime.h: int64_t chelis_list_len ( const chelis_list * list ) ;",
        &["numeric-op"],
        "[05-OP-32]",
        "shape_index(container, parameters...) -> result"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_runtime.h: int64_t chelis_mmap_len ( const chelis_mapped_file * mapped ) ;",
        &["numeric-op"],
        "[05-OP-32]",
        "shape_index(container, parameters...) -> result"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_runtime.h: int64_t chelis_string_len ( chelis_string value ) ;",
        &["numeric-op"],
        "[05-OP-32]",
        "shape_index(container, parameters...) -> result"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_runtime.h: int64_t chelis_tuple_len ( const chelis_tuple * tuple ) ;",
        &["numeric-op"],
        "[05-OP-32]",
        "shape_index(container, parameters...) -> result"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_runtime.h: void chelis_print_adt ( const chelis_adt * adt ) ;",
        &[],
        "[05-OP-32]",
        "shape_index(container, parameters...) -> result"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_runtime.h: void chelis_print_dict ( const chelis_dict * dict ) ;",
        &[],
        "[05-OP-32]",
        "shape_index(container, parameters...) -> result"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_runtime.h: void chelis_print_list ( const chelis_list * list ) ;",
        &[],
        "[05-OP-32]",
        "shape_index(container, parameters...) -> result"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_runtime.h: void chelis_print_tuple ( const chelis_tuple * tuple ) ;",
        &[],
        "[05-OP-32]",
        "shape_index(container, parameters...) -> result"
    ),
    final_numeric_row!(
        "std-adt-numeric",
        "decimal::Decimal: () (variant {} Decimal (field {} coefficient (t-prim {} int64)) (field {} scale (t-prim {} int64)))",
        &["numeric-op"],
        "[05-OP-34]",
        "numeric_adt"
    ),
    final_numeric_row!(
        "std-adt-numeric",
        "io/json::Json: () (variant {} JsonNull) (variant {} JsonBool (t-prim {} bool)) (variant {} JsonInt (t-prim {} int64)) (variant {} JsonBigInt (t-prim {} string)) (variant {} JsonFloat (t-prim {} f64)) (variant {} JsonString (t-prim {} string)) (variant {} JsonArray (t-adt {} List (t-adt {} Json))) (variant {} JsonObject (t-adt {} Dict (t-prim {} string) (t-adt {} Json)))",
        &["float-carrier", "numeric-op"],
        "[05-OP-34]",
        "numeric_adt"
    ),
    final_numeric_row!(
        "std-adt-numeric",
        "time::Date: () (variant {} Date (field {} year (t-prim {} int64)) (field {} month (t-prim {} int64)) (field {} day (t-prim {} int64)))",
        &["numeric-op"],
        "[05-OP-34]",
        "numeric_adt"
    ),
    final_numeric_row!(
        "std-adt-numeric",
        "time::Duration: () (variant {} Duration (field {} days (t-prim {} int64)) (field {} hours (t-prim {} int64)) (field {} minutes (t-prim {} int64)) (field {} seconds (t-prim {} int64)))",
        &["numeric-op"],
        "[05-OP-34]",
        "numeric_adt"
    ),
    final_numeric_row!(
        "std-adt-numeric",
        "tokenizer::Tokenizer: () (variant {} BpeTokenizer (t-adt {} Dict (t-prim {} string) (t-prim {} int64)) (t-adt {} Dict (t-prim {} string) (t-prim {} int64)) (t-adt {} Dict (t-prim {} int64) (t-prim {} string)) (t-prim {} int64))",
        &["numeric-op"],
        "[05-OP-34]",
        "numeric_adt"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "contracts::normal_cdf: [p: Float] (t-fn {} (t-var {} p) (t-var {} p))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "contracts::normal_cdf_contract_samples: (t-fn {} (t-prim {} int64))",
        &["numeric-op"],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "contracts::normal_cdf_contract_seed: (t-fn {} (t-prim {} int64))",
        &["numeric-op"],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "contracts::standard_contract_tolerance: (t-fn {} (t-prim {} f32))",
        &["float-carrier"],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "decimal::decimal: (t-fn {} (t-prim {} string) (t-adt {} Decimal))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "decimal::decimal_add: (t-fn {} (t-adt {} Decimal) (t-adt {} Decimal) (t-adt {} Decimal))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "decimal::decimal_div: (t-fn {} (t-adt {} Decimal) (t-adt {} Decimal) (t-prim {} int64) (t-adt {} RoundingMode) (t-adt {} Decimal))",
        &["numeric-op"],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "decimal::decimal_eq: (t-fn {} (t-adt {} Decimal) (t-adt {} Decimal) (t-prim {} bool))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "decimal::decimal_from_int: (t-fn {} (t-prim {} int64) (t-adt {} Decimal))",
        &["numeric-op"],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "decimal::decimal_gt: (t-fn {} (t-adt {} Decimal) (t-adt {} Decimal) (t-prim {} bool))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "decimal::decimal_gte: (t-fn {} (t-adt {} Decimal) (t-adt {} Decimal) (t-prim {} bool))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "decimal::decimal_lt: (t-fn {} (t-adt {} Decimal) (t-adt {} Decimal) (t-prim {} bool))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "decimal::decimal_lte: (t-fn {} (t-adt {} Decimal) (t-adt {} Decimal) (t-prim {} bool))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "decimal::decimal_mul: (t-fn {} (t-adt {} Decimal) (t-adt {} Decimal) (t-adt {} Decimal))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "decimal::decimal_sub: (t-fn {} (t-adt {} Decimal) (t-adt {} Decimal) (t-adt {} Decimal))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "decimal::decimal_to_float: (t-fn {} (t-adt {} Decimal) (t-prim {} f64))",
        &["float-carrier"],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "decimal::decimal_to_string: (t-fn {} (t-adt {} Decimal) (t-prim {} string))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "decimal::try_decimal: (t-fn {} (t-prim {} string) (t-adt {} Option (t-adt {} Decimal)))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "index::drop_list: (t-fn {} (t-adt {} List (t-var {} item)) (t-prim {} int64) (t-adt {} List (t-var {} item)))",
        &["numeric-op"],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "index::list_index: (t-fn {} (t-adt {} List (t-var {} item)) (t-prim {} int64) (t-var {} item))",
        &["numeric-op"],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "index::take_list: (t-fn {} (t-adt {} List (t-var {} item)) (t-prim {} int64) (t-adt {} List (t-var {} item)))",
        &["numeric-op"],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "init/kaiming::kaiming_normal: [p: Float] (t-fn {eff: (effects {} random)} (t-ref {} (t-tensor {} (d-rank {} r) (t-var {} p))) (t-var {} p) (t-tensor {} (d-rank {} r) (t-var {} p)))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "init/kaiming::kaiming_uniform: [p: Float] (t-fn {eff: (effects {} random)} (t-ref {} (t-tensor {} (d-rank {} r) (t-var {} p))) (t-var {} p) (t-tensor {} (d-rank {} r) (t-var {} p)))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "init/random::normal_like: [p: Float] (t-fn {eff: (effects {} random)} (t-ref {} (t-tensor {} (d-rank {} r) (t-var {} p))) (t-var {} p) (t-var {} p) (t-tensor {} (d-rank {} r) (t-var {} p)))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "init/xavierext::trunc_normal: [p: Float] (t-fn {eff: (effects {} random)} (t-ref {} (t-tensor {} (d-rank {} r) (t-var {} p))) (t-var {} p) (t-var {} p) (t-var {} p) (t-var {} p) (t-tensor {} (d-rank {} r) (t-var {} p)))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "init/xavierext::xavier_normal: [p: Float] (t-fn {eff: (effects {} random)} (t-ref {} (t-tensor {} (d-rank {} r) (t-var {} p))) (t-var {} p) (t-var {} p) (t-tensor {} (d-rank {} r) (t-var {} p)))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "init/xavierext::xavier_uniform: [p: Float] (t-fn {eff: (effects {} random)} (t-ref {} (t-tensor {} (d-rank {} r) (t-var {} p))) (t-var {} p) (t-var {} p) (t-tensor {} (d-rank {} r) (t-var {} p)))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "io/json::json_array: (t-fn {} (t-adt {} Option (t-adt {} Json)) (t-adt {} Option (t-adt {} List (t-adt {} Json))))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "io/json::json_bigint: (t-fn {} (t-adt {} Option (t-adt {} Json)) (t-adt {} Option (t-prim {} string)))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "io/json::json_bool: (t-fn {} (t-adt {} Option (t-adt {} Json)) (t-adt {} Option (t-prim {} bool)))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "io/json::json_float: (t-fn {} (t-adt {} Option (t-adt {} Json)) (t-adt {} Option (t-prim {} f64)))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "io/json::json_get: (t-fn {} (t-adt {} Json) (t-prim {} string) (t-adt {} Option (t-adt {} Json)))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "io/json::json_int: (t-fn {} (t-adt {} Option (t-adt {} Json)) (t-adt {} Option (t-prim {} int64)))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "io/json::json_is_null: (t-fn {} (t-adt {} Option (t-adt {} Json)) (t-prim {} bool))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "io/json::json_object: (t-fn {} (t-adt {} Option (t-adt {} Json)) (t-adt {} Option (t-adt {} Dict (t-prim {} string) (t-adt {} Json))))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "io/json::json_string: (t-fn {} (t-adt {} Option (t-adt {} Json)) (t-adt {} Option (t-prim {} string)))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "io/json::load_json: (t-fn {eff: (effects {} io)} (t-prim {} string) (t-adt {} Json))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "io/json::parse_json: (t-fn {} (t-prim {} string) (t-adt {} Json))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "io/json::to_json: (t-fn {} (t-adt {} Json) (t-prim {} string))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "io/json::try_load_json: (t-fn {eff: (effects {} io)} (t-prim {} string) (t-adt {} Option (t-adt {} Json)))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "io/json::try_parse_json: (t-fn {} (t-prim {} string) (t-adt {} Option (t-adt {} Json)))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "io/json::try_to_json: (t-fn {} (t-adt {} Json) (t-adt {} Option (t-prim {} string)))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "io/json::try_write_json: (t-fn {eff: (effects {} io)} (t-prim {} string) (t-adt {} Json) (t-adt {} Option (t-unit {})))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "io/json::write_json: (t-fn {eff: (effects {} io)} (t-prim {} string) (t-adt {} Json) (t-unit {}))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "io::mmap_size: (t-fn {eff: (effects {} io)} (t-prim {} string) (t-prim {} int64))",
        &["numeric-op"],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "io::read_head_bytes: (t-fn {eff: (effects {} io)} (t-prim {} string) (t-prim {} int64) (t-adt {} List (t-prim {} int64)))",
        &["numeric-op"],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "process::run: (t-fn {eff: (effects {} io)} (t-prim {} string) (t-adt {} List (t-prim {} string)) (t-tuple {} (t-prim {} int64) (t-prim {} string) (t-prim {} string)))",
        &["numeric-op"],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "process::run_chelis: (t-fn {eff: (effects {} io)} (t-adt {} List (t-prim {} string)) (t-tuple {} (t-prim {} int64) (t-prim {} string) (t-prim {} string)))",
        &["numeric-op"],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "scalar::abs: [p: Numeric] (t-fn {} (t-var {} p) (t-var {} p))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "scalar::max: [p: Numeric] (t-fn {} (t-var {} p) (t-var {} p) (t-var {} p))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "scalar::min: [p: Numeric] (t-fn {} (t-var {} p) (t-var {} p) (t-var {} p))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "sort::sort: [p: Numeric] (t-fn {} (t-ref {} (t-tensor {} (d-rank {} r) (t-var {} p))) (t-prim {} int32) (t-tuple {} (t-tensor {} (d-rank {} r) (t-var {} p)) (t-tensor {} (d-rank {} r) (t-prim {} int64))))",
        &["numeric-op"],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "tensor/construct::arange: [p: Int] (t-fn {} (t-var {} p) (t-var {} p) (t-tensor {} (d-var {} n) (t-var {} p)))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "tensor/construct::linspace: [p: Float] (t-fn {} (t-var {} p) (t-var {} p) (t-prim {} int64) (t-tensor {} (d-var {} n) (t-var {} p)))",
        &["numeric-op"],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "tensor/construct::squeeze: (t-fn {} (t-ref {} (t-tensor {} (d-rank {} pre) (d-lit {} 1) (d-rank {} post) (t-var {} p))) (t-prim {} int32) (t-tensor {} (d-rank {} pre) (d-rank {} post) (t-var {} p)))",
        &["numeric-op"],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "tensor/construct::stack: (t-fn {} (t-adt {} List (t-tensor {} (d-rank {} pre) (d-rank {} post) (t-var {} p))) (t-prim {} int32) (t-tensor {} (d-rank {} pre) (d-name {} rows) (d-rank {} post) (t-var {} p)))",
        &["numeric-op"],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "tensor/construct::unsqueeze: (t-fn {} (t-ref {} (t-tensor {} (d-rank {} pre) (d-rank {} post) (t-var {} p))) (t-prim {} int32) (t-tensor {} (d-rank {} pre) (d-lit {} 1) (d-rank {} post) (t-var {} p)))",
        &["numeric-op"],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "tensor/mask::where_indices: (t-fn {} (t-ref {} (t-tensor {} (d-rank {} r) (t-prim {} bool))) (t-tensor {} (d-var {} hits) (t-prim {} int64)))",
        &["numeric-op"],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "test::assert_close: [p: Float] (t-fn {eff: (effects {} test)} (t-var {} p) (t-var {} p) (t-var {} p) (t-prim {} string) (t-unit {}))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "test::assert_close_tensor: [p: Float] (t-fn {eff: (effects {} test)} (t-ref {} (t-tensor {} (d-rank {} r) (t-var {} p))) (t-ref {} (t-tensor {} (d-rank {} r) (t-var {} p))) (t-var {} p) (t-prim {} string) (t-unit {}))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "test::assert_eq: (t-fn {eff: (effects {} test)} (t-var {} q) (t-var {} q) (t-prim {} string) (t-unit {}))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "test::assert_eq_tensor: (t-fn {eff: (effects {} test)} (t-ref {} (t-tensor {} (d-rank {} r) (t-var {} p))) (t-ref {} (t-tensor {} (d-rank {} r) (t-var {} p))) (t-prim {} string) (t-unit {}))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "test::assert_shape: (t-fn {eff: (effects {} test)} (t-ref {} (t-tensor {} (d-rank {} r) (t-var {} p))) (t-adt {} List (t-prim {} int64)) (t-prim {} string) (t-unit {}))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "time::add_days: (t-fn {} (t-adt {} Date) (t-prim {} int64) (t-adt {} Date))",
        &["numeric-op"],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "time::date: (t-fn {} (t-prim {} int64) (t-prim {} int64) (t-prim {} int64) (t-adt {} Date))",
        &["numeric-op"],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "time::date_gt: (t-fn {} (t-adt {} Date) (t-adt {} Date) (t-prim {} bool))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "time::date_gte: (t-fn {} (t-adt {} Date) (t-adt {} Date) (t-prim {} bool))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "time::date_lt: (t-fn {} (t-adt {} Date) (t-adt {} Date) (t-prim {} bool))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "time::date_lte: (t-fn {} (t-adt {} Date) (t-adt {} Date) (t-prim {} bool))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "time::date_to_string: (t-fn {} (t-adt {} Date) (t-prim {} string))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "time::day_of_week: (t-fn {} (t-adt {} Date) (t-adt {} DayOfWeek))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "time::day_of_week_name: (t-fn {} (t-adt {} Date) (t-prim {} string))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "time::day_of_year: (t-fn {} (t-adt {} Date) (t-prim {} int64))",
        &["numeric-op"],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "time::days_between: (t-fn {} (t-adt {} Date) (t-adt {} Date) (t-prim {} int64))",
        &["numeric-op"],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "time::duration: (t-fn {} (t-prim {} int64) (t-prim {} int64) (t-prim {} int64) (t-prim {} int64) (t-adt {} Duration))",
        &["numeric-op"],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "time::is_leap_year: (t-fn {} (t-prim {} int64) (t-prim {} bool))",
        &["numeric-op"],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "time::parse_date: (t-fn {} (t-prim {} string) (t-adt {} Option (t-adt {} Date)))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "time::sub_days: (t-fn {} (t-adt {} Date) (t-prim {} int64) (t-adt {} Date))",
        &["numeric-op"],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "time::try_date: (t-fn {} (t-prim {} int64) (t-prim {} int64) (t-prim {} int64) (t-adt {} Option (t-adt {} Date)))",
        &["numeric-op"],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "tokenizer::batch_encode: (t-fn {} (t-adt {} Tokenizer) (t-adt {} List (t-prim {} string)) (t-prim {} int64) (t-prim {} int64) (t-tensor {} (d-name {} batch) (d-name {} seq) (t-prim {} int64)))",
        &["numeric-op"],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "tokenizer::decode: (t-fn {} (t-adt {} Tokenizer) (t-adt {} List (t-prim {} int64)) (t-prim {} string))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "tokenizer::encode: (t-fn {} (t-adt {} Tokenizer) (t-prim {} string) (t-adt {} List (t-prim {} int64)))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "tokenizer::load_tokenizer: (t-fn {eff: (effects {} io)} (t-prim {} string) (t-adt {} Tokenizer))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    final_numeric_row!(
        "std-def-numeric",
        "tokenizer::try_load_tokenizer: (t-fn {eff: (effects {} io)} (t-prim {} string) (t-adt {} Option (t-adt {} Tokenizer)))",
        &[],
        "[05-OP-35]",
        "stdlib_numeric_def"
    ),
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_dict * chelis_dict_merge ( const chelis_dict * left , const chelis_dict * right ) ;",
            &[],
        ),
        atom: "[05-OP-32]",
        authority_anchor: "`dict_merge` process entries from left to right",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_option * chelis_dict_get ( const chelis_dict * dict , chelis_value key ) ;",
            &[],
        ),
        atom: "[05-OP-32]",
        authority_anchor: "`dict_get` returns one owned [05-OP-44] option\n> node that is `None` only for absence",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_list * chelis_tensor_elements ( const chelis_tensor * tensor ) ;",
            &[],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_elements` boxes every\n> element as its exact scalar in row-major order",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_list * chelis_tensor_split ( const chelis_tensor * tensor , int32_t axis , const chelis_list * sizes ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "nonnegative int64 sizes whose checked sum equals the selected extent",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_option * chelis_dict_get_scalar ( const chelis_dict * dict , chelis_value key , chelis_dtype dtype ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-31]",
        authority_anchor: "Dictionary lookup admits only [05-OP-32]'s key domain",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_option * chelis_parse_scalar ( chelis_string text , chelis_dtype dtype ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-31]",
        authority_anchor: "parsing accepts a strict\n> decimal superset",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_scalar chelis_scalar_from_bits ( chelis_dtype dtype , uint64_t bits ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-31]",
        authority_anchor: "the exact stored image and all unused high bits are zero",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_scalar chelis_tensor_to_scalar ( const chelis_tensor * tensor ) ;",
            &[],
        ),
        atom: "[05-OP-31]",
        authority_anchor: "Tensor extraction requires a rank-zero tensor with exactly\n> one element",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_scalar chelis_value_unbox_scalar ( chelis_value value ) ;",
            &[],
        ),
        atom: "[05-OP-31]",
        authority_anchor: "Every consumer validates both the foreign dtype value",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_string chelis_string_from_scalar ( chelis_scalar value ) ;",
            &[],
        ),
        atom: "[05-OP-31]",
        authority_anchor: "Rendering follows [05-OBS-1..2] at the\n> scalar's own dtype",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_tensor * chelis_alloc ( int32_t rank , const int64_t * shape , chelis_dtype dtype ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "Allocation returns owned, contiguous, row-major, zero-filled storage",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_tensor * chelis_tensor_entry_borrow ( int32_t rank , const int64_t * shape , chelis_dtype dtype , const void * data , int64_t byte_capacity ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-44]",
        authority_anchor: "`chelis_tensor_entry_borrow` validates the declared rank,\n> extents, dtype, alignment, and capacity",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_read_view chelis_tensor_read_view ( const chelis_tensor * tensor ) ;",
            &[],
        ),
        atom: "[05-OP-44]",
        authority_anchor: "`chelis_tensor_read_view` returns\n> [05-OP-31]'s read view",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_write_view chelis_tensor_write_view ( const chelis_tensor_write * guard ) ;",
            &[],
        ),
        atom: "[05-OP-44]",
        authority_anchor: "`chelis_tensor_write_view` borrows its `const` guard",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_tensor * chelis_contiguous ( const chelis_tensor * tensor ) ;",
            &[],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`contiguous` preserves every element's\n> exact stored bits in row-major order",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_tensor * chelis_pad_sequences ( const chelis_list * sequences , chelis_scalar pad_value ) ;",
            &[],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "Padding follows [05-OP-9..10] exactly",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_tensor * chelis_pad_sequences_to ( const chelis_list * sequences , int64_t width , chelis_scalar pad_value ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "Padding follows [05-OP-9..10] exactly",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_tensor * chelis_scalar_tensor ( chelis_scalar value ) ;",
            &[],
        ),
        atom: "[05-OP-31]",
        authority_anchor: "empty product equal to one",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_tensor * chelis_tensor_clamp ( const chelis_tensor * tensor , const chelis_tensor * lower , const chelis_tensor * upper ) ;",
            &[],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`clamp` admits signed-integer and float tensors. Each bound has the input",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_tensor * chelis_tensor_cmplt ( const chelis_tensor * left , const chelis_tensor * right ) ;",
            &[],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`cmplt` requires identical shapes and identical active signed-integer or\n> float dtypes, compares stored values without conversion",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_tensor * chelis_tensor_concat ( const chelis_list * parts , int32_t axis ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`concat` requires a nonempty list of tensors with one common rank and dtype",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_tensor * chelis_tensor_cumsum ( const chelis_tensor * tensor , int32_t axis ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`cumsum` admits signed-integer and float tensors and returns\n> `sum_result(p, default(p))` at the input shape",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_tensor * chelis_tensor_diagonal ( const chelis_tensor * tensor , int32_t axis1 , int32_t axis2 ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`diagonal` admits every active dtype including bool, requires distinct axes",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_tensor * chelis_tensor_einsum ( chelis_string equation , const chelis_tensor * left , const chelis_tensor * right , chelis_dtype accumulator ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`einsum` accepts exactly the grammar `[a-z]*,[a-z]*->[a-z]*`",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_tensor * chelis_tensor_from_values ( const chelis_list * list , chelis_dtype dtype ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "nested list whose scalar leaves all have exactly the requested dtype",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_tensor * chelis_tensor_gather ( const chelis_tensor * tensor , const chelis_tensor * indices , int32_t axis ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`gather` admits an index tensor of any active signed-integer dtype",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_tensor * chelis_tensor_scatter_add ( const chelis_tensor * base , const chelis_tensor * indices , const chelis_tensor * updates , int32_t axis ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "Add starts each\n> destination's leaf sequence with the base value",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_tensor * chelis_tensor_scatter_replace ( const chelis_tensor * base , const chelis_tensor * indices , const chelis_tensor * updates , int32_t axis ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "§3.5's row-major last-write-wins rule",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_tensor * chelis_tensor_trace ( const chelis_tensor * tensor , int32_t axis1 , int32_t axis2 ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`trace` is that diagonal\n> followed by [05-OP-30]'s canonical balanced tree and default accumulator",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_tensor * chelis_tensor_where ( const chelis_tensor * condition , const chelis_tensor * then_tensor , const chelis_tensor * else_tensor ) ;",
            &[],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "exact public C counterpart `chelis_tensor_where` in that registry",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_tuple * chelis_tensor_sort ( const chelis_tensor * tensor , int32_t axis ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`sort` admits signed-integer and float tensors and returns `(values,\n> indices)`",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_value chelis_value_box_scalar ( chelis_scalar value ) ;",
            &[],
        ),
        atom: "[05-OP-31]",
        authority_anchor: "Scalar\n> embeds the complete canonical `chelis_scalar`",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: int32_t chelis_tensor_rank ( const chelis_tensor * tensor ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "need dimensions use `chelis_tensor_rank`",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: int64_t chelis_dtype_size ( chelis_dtype dtype ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-31]",
        authority_anchor: "`chelis_dtype_size` returns the exact byte width",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: int64_t chelis_tensor_numel ( const chelis_tensor * tensor ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "element counts are `int64_t`",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: int64_t chelis_tensor_shape ( const chelis_tensor * tensor , int32_t axis ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "use `chelis_tensor_rank` and `chelis_tensor_shape`",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: void chelis_fill_scalar ( chelis_tensor_write * guard , chelis_scalar value ) ;",
            &[],
        ),
        atom: "[05-OP-31]",
        authority_anchor: "Fill requires the scalar dtype to equal the tensor dtype and\n> writes the exact scalar bits to every element",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: void chelis_tensor_repurpose ( chelis_tensor * tensor , chelis_scalar rank , const chelis_scalar * shape ) ;",
            &[],
        ),
        atom: "[05-OP-44]",
        authority_anchor: "`chelis_tensor_repurpose` is the one descriptor-metadata mutation",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: int64_t chelis_tensor_stride ( const chelis_tensor * tensor , int32_t axis ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_stride` projects the checked contiguous suffix stride",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: int64_t chelis_tensor_byte_count ( const chelis_tensor * tensor ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_byte_count` projects the checked logical byte",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: int64_t chelis_tensor_elementwise_index_step ( const chelis_tensor * input , const chelis_tensor * domain ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_elementwise_index_step` validates input and domain tensor metadata",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: int64_t chelis_tensor_elementwise_index_step_for_shape ( const chelis_tensor * input , chelis_scalar rank , const chelis_scalar * shape ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_elementwise_index_step_for_shape` applies the same rule",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: void chelis_tensor_unravel_index ( const chelis_tensor * tensor , chelis_scalar index , chelis_scalar * coordinates ) ;",
            &[],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_unravel_index` converts an exact tagged int64 linear index into",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: int64_t chelis_tensor_flat_index ( const chelis_tensor * tensor , const chelis_scalar * coordinates ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_flat_index` converts rank-many exact tagged int64 coordinates",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: void chelis_tensor_check_permute ( const chelis_tensor * tensor , chelis_scalar rank , const chelis_scalar * shape , const chelis_scalar * axes ) ;",
            &[],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_check_permute` and `chelis_tensor_check_expand` validate a target",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: void chelis_tensor_check_expand ( const chelis_tensor * tensor , chelis_scalar rank , const chelis_scalar * shape , int32_t axis ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_check_permute` and `chelis_tensor_check_expand` validate a target",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: void chelis_tensor_pad_shape ( const chelis_tensor * tensor , chelis_scalar rank , const chelis_scalar * before , const chelis_scalar * after , chelis_scalar * shape ) ;",
            &[],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_pad_shape`, `chelis_tensor_shrink_shape`, and",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: void chelis_tensor_shrink_shape ( const chelis_tensor * tensor , chelis_scalar rank , const chelis_scalar * start , const chelis_scalar * end , chelis_scalar * shape ) ;",
            &[],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_pad_shape`, `chelis_tensor_shrink_shape`, and",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: void chelis_tensor_stride_shape ( const chelis_tensor * tensor , chelis_scalar rank , const chelis_scalar * steps , chelis_scalar * shape ) ;",
            &[],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_pad_shape`, `chelis_tensor_shrink_shape`, and",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: int64_t chelis_tensor_affine_index ( const chelis_tensor * tensor , const chelis_scalar * coordinates , const chelis_scalar * offsets , const chelis_scalar * steps ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_affine_index` takes rank-many exact tagged int64 coordinates,",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_tensor * chelis_tensor_alloc_like ( const chelis_tensor * input , chelis_scalar exemplar ) ;",
            &[],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_alloc_like` takes a validated input tensor and a canonical",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: void chelis_tensor_check_literal ( chelis_scalar rank , const chelis_scalar * shape , chelis_scalar exemplar , chelis_scalar count ) ;",
            &[],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_check_literal` validates a complete result shape and literal",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: void chelis_tensor_write_literal ( chelis_tensor_write * guard , chelis_scalar count , const chelis_scalar * values ) ;",
            &[],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_write_literal` takes a live tensor write guard, an exact tagged",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_movement_plan * chelis_tensor_permute_plan ( const chelis_tensor * input , chelis_scalar rank , const chelis_scalar * axes ) ;",
            &[],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_permute_plan`, `chelis_tensor_expand_plan`, and",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_movement_plan * chelis_tensor_expand_plan ( const chelis_tensor * input , chelis_scalar axis , chelis_scalar size , chelis_movement_op operation ) ;",
            &[],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_permute_plan`, `chelis_tensor_expand_plan`, and",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_movement_plan * chelis_tensor_affine_plan ( const chelis_tensor * input , chelis_scalar rank , const chelis_scalar * first , const chelis_scalar * second , chelis_movement_op operation ) ;",
            &[],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_permute_plan`, `chelis_tensor_expand_plan`, and",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: int64_t chelis_movement_extent ( const chelis_movement_plan * plan , chelis_movement_side side , chelis_scalar axis ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_permute_plan`, `chelis_tensor_expand_plan`, and",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: int64_t chelis_movement_count ( const chelis_movement_plan * plan ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_permute_plan`, `chelis_tensor_expand_plan`, and",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: int64_t chelis_movement_index ( const chelis_movement_plan * plan , chelis_scalar linear ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_permute_plan`, `chelis_tensor_expand_plan`, and",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: void chelis_movement_check_target ( const chelis_movement_plan * plan , chelis_scalar rank , const chelis_scalar * shape ) ;",
            &[],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_permute_plan`, `chelis_tensor_expand_plan`, and",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: void chelis_movement_plan_release ( chelis_movement_plan * plan ) ;",
            &[],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_permute_plan`, `chelis_tensor_expand_plan`, and",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_window_plan * chelis_tensor_window_plan ( const chelis_tensor * input , chelis_scalar count , const chelis_scalar * window , const chelis_scalar * steps , chelis_window_op operation ) ;",
            &[],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_window_plan` snapshots the complete input and valid-padding",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: int64_t chelis_window_extent ( const chelis_window_plan * plan , chelis_window_side side , chelis_scalar axis ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_window_plan` snapshots the complete input and valid-padding",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: int64_t chelis_window_count ( const chelis_window_plan * plan ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_window_plan` snapshots the complete input and valid-padding",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: int64_t chelis_window_index ( const chelis_window_plan * plan , chelis_scalar group , chelis_scalar leaf ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_window_plan` snapshots the complete input and valid-padding",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: void chelis_window_check_tensor ( const chelis_window_plan * plan , const chelis_tensor * tensor , chelis_window_side side ) ;",
            &[],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_window_plan` snapshots the complete input and valid-padding",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: void chelis_window_check_target ( const chelis_window_plan * plan , chelis_window_side side , chelis_scalar rank , const chelis_scalar * shape ) ;",
            &[],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_window_plan` snapshots the complete input and valid-padding",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: void chelis_window_plan_release ( chelis_window_plan * plan ) ;",
            &[],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_window_plan` snapshots the complete input and valid-padding",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_matmul_plan * chelis_tensor_matmul_plan ( const chelis_tensor * left , const chelis_tensor * right , chelis_scalar exemplar ) ;",
            &[],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_matmul_plan` snapshots two checked row-major matrix operands",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: int64_t chelis_matmul_extent ( const chelis_matmul_plan * plan , chelis_scalar axis ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_matmul_plan` snapshots two checked row-major matrix operands",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: int64_t chelis_matmul_dimension ( const chelis_matmul_plan * plan , chelis_matmul_dimension_kind dimension ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_matmul_plan` snapshots two checked row-major matrix operands",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: int64_t chelis_matmul_batch_count ( const chelis_matmul_plan * plan ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_matmul_plan` snapshots two checked row-major matrix operands",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: int64_t chelis_matmul_matrix_count ( const chelis_matmul_plan * plan , chelis_matmul_part part ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_matmul_plan` snapshots two checked row-major matrix operands",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: int64_t chelis_matmul_index ( const chelis_matmul_plan * plan , chelis_matmul_part part , chelis_scalar batch , chelis_scalar element ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_matmul_plan` snapshots two checked row-major matrix operands",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: void chelis_matmul_check_target ( const chelis_matmul_plan * plan , chelis_scalar rank , const chelis_scalar * shape ) ;",
            &[],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_matmul_plan` snapshots two checked row-major matrix operands",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: void chelis_matmul_check_scratch ( const chelis_matmul_plan * plan , chelis_matmul_part part , chelis_scalar exemplar ) ;",
            &[],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_matmul_plan` snapshots two checked row-major matrix operands",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: void chelis_matmul_check_vendor ( const chelis_matmul_plan * plan , chelis_scalar maximum ) ;",
            &[],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_matmul_plan` snapshots two checked row-major matrix operands",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: void chelis_matmul_plan_release ( chelis_matmul_plan * plan ) ;",
            &[],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_matmul_plan` snapshots two checked row-major matrix operands",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_sparse_plan * chelis_tensor_sparse_plan ( const chelis_tensor * base , const chelis_tensor * indices , const chelis_tensor * updates , chelis_scalar axis , chelis_sparse_op operation ) ;",
            &[],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_sparse_plan` snapshots checked base, index, and (for scatter)",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: int64_t chelis_sparse_extent ( const chelis_sparse_plan * plan , chelis_scalar axis ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_sparse_plan` snapshots checked base, index, and (for scatter)",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: int64_t chelis_sparse_count ( const chelis_sparse_plan * plan ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_sparse_plan` snapshots checked base, index, and (for scatter)",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: int64_t chelis_sparse_index_slot ( const chelis_sparse_plan * plan , chelis_scalar linear ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_sparse_plan` snapshots checked base, index, and (for scatter)",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: int64_t chelis_sparse_data_index ( const chelis_sparse_plan * plan , chelis_scalar linear , chelis_scalar selected ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_sparse_plan` snapshots checked base, index, and (for scatter)",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: void chelis_sparse_check_target ( const chelis_sparse_plan * plan , chelis_scalar rank , const chelis_scalar * shape ) ;",
            &[],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_sparse_plan` snapshots checked base, index, and (for scatter)",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: void chelis_sparse_plan_release ( chelis_sparse_plan * plan ) ;",
            &[],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_sparse_plan` snapshots checked base, index, and (for scatter)",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_reduction_plan * chelis_tensor_reduction_plan ( const chelis_tensor * tensor , chelis_scalar axis_count , const chelis_scalar * axes , chelis_scalar exemplar , chelis_reduction_op operation ) ;",
            &[],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_reduction_plan` snapshots checked tensor metadata;",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_reduction_plan * chelis_shape_reduction_plan ( chelis_scalar rank , const chelis_scalar * shape , chelis_scalar axis_count , const chelis_scalar * axes , chelis_scalar exemplar , chelis_reduction_op operation ) ;",
            &[],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_reduction_plan` snapshots checked tensor metadata;",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: int64_t chelis_reduction_count ( const chelis_reduction_plan * plan ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_reduction_plan` snapshots checked tensor metadata;",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: int64_t chelis_reduction_extent ( const chelis_reduction_plan * plan , chelis_scalar axis ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_reduction_plan` snapshots checked tensor metadata;",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: int64_t chelis_reduction_index ( const chelis_reduction_plan * plan , chelis_scalar outer , chelis_scalar leaf ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_reduction_plan` snapshots checked tensor metadata;",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: void chelis_reduction_check_target ( const chelis_reduction_plan * plan , chelis_scalar rank , const chelis_scalar * shape ) ;",
            &[],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_reduction_plan` snapshots checked tensor metadata;",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: void chelis_reduction_check_scratch ( const chelis_reduction_plan * plan , chelis_scalar exemplar ) ;",
            &[],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_reduction_plan` snapshots checked tensor metadata;",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: void chelis_reduction_plan_release ( chelis_reduction_plan * plan ) ;",
            &[],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_reduction_plan` snapshots checked tensor metadata;",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: void chelis_tensor_check_reshape ( const chelis_tensor * tensor , chelis_scalar rank , const chelis_scalar * shape ) ;",
            &[],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_check_reshape` takes rank and every target extent as exact tagged",
    },
    FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "chelis_runtime.h: chelis_tensor * chelis_tensor_reshape ( const chelis_tensor * tensor , const chelis_list * shape ) ;",
            &[],
        ),
        atom: "[05-OP-33]",
        authority_anchor: "`chelis_tensor_reshape` accepts a live, flat `List<int64>`",
    },
];

const BACKEND_TAGGED_TRANSPORT_ROWS: &[StaticSurfaceDescriptor] = &[StaticSurfaceDescriptor::new(
    PRIMARY_CENSUS_FAMILY,
    "header-struct",
    "chelis_device_descriptor.h: typedef struct { void * data ; const int64_t * shape ; const int64_t * strides ; int64_t count ; int64_t byte_capacity ; int32_t rank ; chelis_dtype dtype ; uint8_t ownership ; uint8_t reserved [ 2 ] ; } chelis_gpu_tensor",
    &["numeric-op"],
)];

const BACKEND_NUMERIC_OPERATION_ROWS: &[FinalNumericOperationRegistration] = &[
    final_numeric_row!(
        "header-export",
        "chelis_device_owner.h: chelis_device_tensor_owner * chelis_device_tensor_alloc ( chelis_metadata_plan * plan ) ;",
        &[],
        "[05-OP-33]",
        "`chelis_gpu_tensor` observation. `chelis_device_tensor_alloc` consumes one"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_device_owner.h: chelis_device_tensor_owner * chelis_device_tensor_borrow ( chelis_metadata_plan * plan , void * data , chelis_scalar byte_capacity ) ;",
        &[],
        "[05-OP-33]",
        "`chelis_device_tensor_borrow` consumes one live metadata plan, validates its"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_device_owner.h: chelis_device_tensor_owner * chelis_device_tensor_import ( const chelis_gpu_tensor * packet ) ;",
        &[],
        "[05-OP-33]",
        "`chelis_device_tensor_import` validates every field of a generated raw packet"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_device_owner.h: const chelis_gpu_tensor * chelis_device_tensor_view ( const chelis_device_tensor_owner * owner ) ;",
        &[],
        "[05-OP-33]",
        "`chelis_device_tensor_view` returns a const generated packet observation tied"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_device_owner.h: int32_t chelis_device_tensor_device ( const chelis_device_tensor_owner * owner ) ;",
        &["numeric-op"],
        "[05-OP-33]",
        "`chelis_device_tensor_device` returns the owner's exact nonnegative int32"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_device_owner.h: chelis_device_tensor_owner * chelis_device_tensor_clone ( const chelis_device_tensor_owner * source ) ;",
        &[],
        "[05-OP-33]",
        "`chelis_device_tensor_clone` returns a distinct owner with a new contiguous"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_device_owner.h: void chelis_device_tensor_release ( chelis_device_tensor_owner * owner ) ;",
        &[],
        "[05-OP-33]",
        "`chelis_device_tensor_release` consumes one live opaque owner exactly once,"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_device_owner.h: void chelis_device_tensor_copy_from_host ( chelis_device_tensor_owner * destination , const chelis_tensor * source ) ;",
        &[],
        "[05-OP-33]",
        "`chelis_device_tensor_copy_from_host` and `chelis_device_tensor_copy_to_host`"
    ),
    final_numeric_row!(
        "header-export",
        "chelis_device_owner.h: void chelis_device_tensor_copy_to_host ( chelis_tensor_write * destination , const chelis_device_tensor_owner * source ) ;",
        &[],
        "[05-OP-33]",
        "`chelis_device_tensor_copy_from_host` and `chelis_device_tensor_copy_to_host`"
    ),
];

fn final_authority_registries() -> AuthorityRegistries<'static> {
    AuthorityRegistries {
        nonnumeric: FINAL_NONNUMERIC_ROWS,
        tagged_transports: FINAL_TAGGED_TRANSPORT_ROWS,
        numeric_operations: FINAL_NUMERIC_OPERATION_ROWS,
    }
}

fn backend_authority_registries() -> AuthorityRegistries<'static> {
    AuthorityRegistries {
        nonnumeric: &[],
        tagged_transports: BACKEND_TAGGED_TRANSPORT_ROWS,
        numeric_operations: BACKEND_NUMERIC_OPERATION_ROWS,
    }
}

fn authority_surface(row: &Row) -> SurfaceDescriptor {
    SurfaceDescriptor {
        family: PRIMARY_CENSUS_FAMILY.to_string(),
        kind: row.kind.clone(),
        id: row.id.clone(),
        flags: row.flags.clone(),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct CoveredLeg {
    leg: String,
    artifact: String,
    enumerator: String,
    command: String,
    expected_success: String,
    mutations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct DeferredLeg {
    leg: String,
    owner: String,
    artifact: String,
    enumerator: String,
    command: String,
    expected_success: String,
    mutations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct CoverageManifest {
    covered: Vec<CoveredLeg>,
    deferred: Vec<DeferredLeg>,
}

fn coverage_manifest() -> CoverageManifest {
    CoverageManifest {
        covered: vec![
            CoveredLeg {
                leg: "header-export".to_string(),
                artifact: "crates/chelis-runtime/include/*.h published closure".to_string(),
                enumerator: "preprocessed_headers -> header_rows".to_string(),
                command: "cargo nextest run -p chelis-cli --test capacity_census_tripwire"
                    .to_string(),
                expected_success: "capacity_census_matches_public_surface passes".to_string(),
                mutations: vec![
                    "reviewer_preprocessor_capacity_seam_is_visible".to_string(),
                    "c_identity_is_token_canonical_across_preprocessor_whitespace".to_string(),
                    "c_identity_is_stable_across_bool_preprocessor_spellings".to_string(),
                ],
            },
            CoveredLeg {
                leg: "header-struct".to_string(),
                artifact: "crates/chelis-runtime/include/*.h published closure".to_string(),
                enumerator: "preprocessed_headers -> header_rows".to_string(),
                command: "cargo nextest run -p chelis-cli --test capacity_census_tripwire"
                    .to_string(),
                expected_success: "capacity_census_matches_public_surface passes".to_string(),
                mutations: vec!["planted_struct_layout_is_inventoried".to_string()],
            },
            CoveredLeg {
                leg: "std-adt-numeric".to_string(),
                artifact: "packages/chelis-std/src/**/*.ch desugared Deep AST".to_string(),
                enumerator: "stdlib_rows -> Reef declaration linker -> resolved type fixed point".to_string(),
                command: "cargo nextest run -p chelis-cli --test capacity_census_tripwire"
                    .to_string(),
                expected_success: "capacity_census_matches_public_surface passes".to_string(),
                mutations: vec![
                    "std_adt_identity_changes_when_same_dtype_variant_changes".to_string(),
                    "stdlib_closure_tests::mutual_nominal_recursion_propagates_numeric_fields".to_string(),
                ],
            },
            CoveredLeg {
                leg: "std-def-numeric".to_string(),
                artifact: "packages/chelis-std/src/**/*.ch desugared Deep AST".to_string(),
                enumerator: "stdlib_rows -> Reef declaration linker -> resolved type fixed point".to_string(),
                command: "cargo nextest run -p chelis-cli --test capacity_census_tripwire"
                    .to_string(),
                expected_success: "capacity_census_matches_public_surface passes".to_string(),
                mutations: vec![
                    "exported_public_numeric_stdlib_def_is_enumerated".to_string(),
                    "stdlib_closure_tests::private_helpers_and_transparent_aliases_survive_relocation".to_string(),
                    "stdlib_closure_tests::unresolved_nominals_and_wrong_nominal_arguments_fail_closed".to_string(),
                ],
            },
            CoveredLeg {
                leg: "prelude-adt-numeric".to_string(),
                artifact: "chelis_types::builtins::register_prelude_adts Rust-registered \
                           prelude value ADTs"
                    .to_string(),
                enumerator: "prelude_adt_rows -> chelis_types::prelude_adt_defs".to_string(),
                command: "cargo nextest run -p chelis-cli --test capacity_census_tripwire"
                    .to_string(),
                expected_success: "capacity_census_matches_public_surface passes".to_string(),
                mutations: vec![
                    "planted_prelude_adt_with_f64_variant_is_detected".to_string(),
                    "final_surface_has_no_duplicate_prelude_json".to_string(),
                ],
            },
            CoveredLeg {
                leg: "backend-runtime-headers".to_string(),
                artifact: "complete chelis_hip_runtime.h and chelis_metal_runtime.h published closures with committed SDK fixtures".to_string(),
                enumerator: "backend_headers::scan -> shared attributed preprocess_root -> header_rows".to_string(),
                command: "cargo nextest run -p chelis-cli --test capacity_census_tripwire -E 'test(backend_headers::)'".to_string(),
                expected_success: "backend_runtime_headers_match_the_reviewed_final_authority passes".to_string(),
                mutations: vec![
                    "backend_roots_cannot_omit_an_unreached_generated_header".to_string(),
                    "backend_environments_cannot_change_a_public_numeric_signature".to_string(),
                    "generated_device_descriptor_requires_exact_tagged_transport_authority".to_string(),
                    "device_owner_callables_require_exact_op33_authority".to_string(),
                ],
            },
            CoveredLeg {
                leg: "wire-schema-numeric-fields".to_string(),
                artifact: "compiler/Python serialization publication roots and their reachable \
                           rustdoc JSON type graph"
                    .to_string(),
                enumerator: "capacity_census_typed.py wire -> private wire verifier: \
                             publication graph inventory, final authority, and execution receipts"
                    .to_string(),
                command: "cargo nextest run -p chelis-compiler-api --test capacity_census_wire"
                    .to_string(),
                expected_success: "wire_schema_numeric_fields_match_the_reviewed_baseline passes"
                    .to_string(),
                mutations: vec![
                    "adding_or_removing_a_public_serialized_f64_field_changes_the_census"
                        .to_string(),
                    "verified_wire_authority_cannot_be_replaced_by_a_descriptor_or_baseline"
                        .to_string(),
                    "wire_baseline_rejects_exception_fields_and_nonfinal_classes".to_string(),
                    "wire_rows_cannot_erase_capacity_or_duplicate_an_identity".to_string(),
                ],
            },
            CoveredLeg {
                leg: "binding-raw-dtype-params".to_string(),
                artifact: "crates/chelis-python/src/lib.rs registered PyO3 callables".to_string(),
                enumerator:
                    "live PyO3 signatures and reachable payloads; all 17 rows require current authority as nine nonnumeric registrations, seven exact tagged transports, or one exact numeric operation"
                        .to_string(),
                command: "cargo nextest run -p chelis-python --test capacity_census_bindings"
                    .to_string(),
                expected_success:
                    "registered_pyfunctions_match_the_reviewed_rustdoc_signatures passes"
                        .to_string(),
                mutations: vec![
                    "a_registered_pyfunction_with_a_raw_dtype_parameter_is_rejected".to_string(),
                    "final_binding_rows_cannot_regain_legacy_admission".to_string(),
                    "copied_missing_and_duplicate_binding_registrations_fail".to_string(),
                ],
            },
        ],
        deferred: vec![],
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Baseline {
    version: u32,
    legs: CoverageManifest,
    rows: Vec<Row>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SemanticRegistration {
    callable: &'static str,
    atom: &'static str,
}

/// Compiler-owned numeric operations which do not land on a discovered
/// capacity-census family. These remain separately pinned until #1294 gives
/// them their own complete enumerator. Discovered rows use the exact shared
/// final-authority registry above instead.
const SEMANTIC_REGISTRATIONS: &[SemanticRegistration] = &[
    SemanticRegistration {
        callable: "[compiler-kernel-numeric] chelis_types::dtype_semantics::PreparedDropout::new(input: &TensorStorage, rate: ScalarValue) -> Result<PreparedDropout, NumericKernelError>",
        atom: "[05-OP-37]",
    },
    SemanticRegistration {
        callable: "[compiler-kernel-numeric] chelis_types::dtype_semantics::PreparedDropout::apply(&self, seed: u64, ordinal: u64) -> Result<TensorStorage, NumericKernelError>",
        atom: "[05-OP-37]",
    },
    SemanticRegistration {
        callable: "[compiler-builtin-numeric] dropout(input: &tensor[D, p_float], rate: p_float) -> tensor[D, p_float]",
        atom: "[05-OP-37]",
    },
    // chelis#759's float-to-integer ladder rung. `cast_trunc` is a
    // compiler-owned numeric callable, so it lands on no enumerated leg
    // (it is neither a C export, an exported stdlib `def`, nor a prelude
    // ADT) and the registration is authored directly against its atom.
    //
    // What this buys, stated honestly: the ATOM direction is enforced by
    // `registration_problem` -- a bogus or nonexistent `[05-OP-N]` fails
    // the tripwire. The PRESENCE of this row is pinned only by
    // `cast_trunc_is_registered_against_its_authority_atom` below.
    // Nothing here structurally prevents a FUTURE compiler-owned op from
    // skipping registration entirely, because no enumerator produces a
    // row for it to be matched against; that gap is the census's
    // off-leg blind spot, not something this entry closes.
    SemanticRegistration {
        callable: "[compiler-builtin-numeric] cast_trunc(source: f16 | bf16 | f32 | f64, \
                   target: int8 | int16 | int32 | int64) -> int8 | int16 | int32 | int64",
        atom: "[05-OP-6]",
    },
    SemanticRegistration {
        callable: "[compiler-builtin-numeric] uniform_like(template: &tensor[D, p], low: f32, \
                   high: f32) -> tensor[D, p]",
        atom: "[05-OP-8]",
    },
    SemanticRegistration {
        callable: "[compiler-builtin-numeric] pad_sequences(sequences: List[List[T]], pad: T) \
                   -> tensor[len(sequences), width, T]",
        atom: "[05-OP-9]",
    },
    SemanticRegistration {
        callable: "[compiler-builtin-numeric] pad_sequences_to(sequences: List[List[T]], \
                   width: int64, pad: T) -> tensor[len(sequences), width, T]",
        atom: "[05-OP-10]",
    },
    SemanticRegistration {
        callable: "[compiler-builtin-numeric] count(input: &tensor[D, bool], axes: int32...) \
                   -> tensor[D\\axes, int64]",
        atom: "[05-OP-29]",
    },
    SemanticRegistration {
        callable: "[compiler-builtin-numeric] relu(input: &tensor[D, p]) -> tensor[D, p]",
        atom: "[05-OP-43]",
    },
    SemanticRegistration {
        callable: "[compiler-risc-numeric] relu_adjoint(input: &tensor[D, p], cotangent: \
                   &tensor[D, p]) -> tensor[D, p]",
        atom: "[05-OP-43]",
    },
];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root resolves")
}

// ---------------------------------------------------------------------------
// Leg A: published header exports and struct layouts
// ---------------------------------------------------------------------------

/// Preprocess every root and merge the per-header outputs. The include
/// closure is followed by the real preprocessor, so an export added to a
/// transitively-included header - or hidden behind a macro - is visible.
fn preprocessed_headers(include_dir: &Path, roots: &[&str]) -> BTreeMap<String, String> {
    preprocessed_headers_with_environment(
        include_dir,
        roots,
        &c_preprocessor::Environment::native_c(),
    )
}

/// Backend headers use explicit C++/Objective-C++ SDK lanes while retaining
/// the primary census's attribution, type-word and context-invariance rules.
fn preprocessed_headers_with_environment(
    include_dir: &Path,
    roots: &[&str],
    environment: &c_preprocessor::Environment<'_>,
) -> BTreeMap<String, String> {
    let sources = header_source_closure(include_dir, roots);
    assert_no_line_directives(&sources);
    assert_context_invariant_headers(include_dir, roots);
    let mut per_file: BTreeMap<String, String> = BTreeMap::new();
    for root in roots {
        let preprocessed = c_preprocessor::preprocess_root(include_dir, root, environment)
            .unwrap_or_else(|error| panic!("{}{}{}", teaching_header(), error, teaching_footer()));
        for (name, text) in preprocessed {
            if let Some(previous) = per_file.get(&name) {
                let previous_rows = header_rows_local(&name, previous);
                let current_rows = header_rows_local(&name, &text);
                assert_eq!(
                    previous_rows,
                    current_rows,
                    "{}CONTEXT-VARYING PUBLIC ABI in `{name}`: two published \
                     roots preprocess it to different exported declarations. \
                     Published ABI must be context-invariant; move the \
                     conditional behind a static implementation detail.{}",
                    teaching_header(),
                    teaching_footer()
                );
            } else {
                per_file.insert(name, text);
            }
        }
    }
    assert_total_attribution(&sources, &per_file);
    assert_roots_reach_every_published_header(include_dir, &sources);
    assert_known_type_words(&per_file);
    per_file
}

/// Invert the classification rule: every TYPE word a published declaration
/// uses must be a spelling the census recognizes, so an unrecognized one
/// fails the build instead of producing an unflagged row.
///
/// This runs on PREPROCESSED text only. Raw sources still carry
/// macro-spelled types (`CHELIS_NUM value`), and those are the
/// context-invariance guard's business, on its own terms, before this one
/// runs.
fn assert_known_type_words(per_file: &BTreeMap<String, String>) {
    let mut typedefs = BTreeMap::new();
    for text in per_file.values() {
        typedefs.append(&mut collect_typedefs(text));
    }
    for (name, text) in per_file {
        for row in header_rows(name, text, &typedefs) {
            let declaration = row.id.split_once(": ").map_or(row.id.as_str(), |(_, d)| d);
            assert_declaration_type_words(name, declaration, &typedefs);
        }
        // A typedef statement is never a row of its own, so the words it
        // introduces would otherwise reach classification only through
        // `resolve_words` - after the positional structure is gone. Checking
        // the statement here is what makes "the word is a local alias" a
        // safe answer at the use site.
        for statement in strip_c_comments(text).split(';') {
            let statement = normalize_ws(statement);
            if statement.starts_with("typedef ") && !statement.contains('{') {
                assert_declaration_type_words(name, &statement, &typedefs);
            }
        }
    }
}

/// A declaration's identifier tokens split into type words and declarator
/// names by POSITION: an identifier followed by another identifier or `*`
/// is a type word, and an identifier followed by `(`, `)`, `,`, `[`, `}`,
/// `;` or the end of the declaration is the thing being declared. A word in
/// either census list counts as a type word wherever it appears, so an
/// abstract parameter (`int f(double);`) cannot hide in name position.
///
/// A TYPE word must be numeric, non-numeric-but-known, a local typedef
/// alias (whose own statement `assert_known_type_words` checks by this same
/// rule), or `chelis_`-prefixed - the project's own opaque handles, whose
/// layouts are inventoried as their own `header-struct` rows.
///
/// A DECLARATOR NAME must not look like a type spelling: a leading `_` is
/// the implementation-reserved identifier namespace and a `_t` suffix is
/// the type-alias convention, so either one in name position means the
/// positional read was wrong and a type word is escaping unclassified.
fn assert_declaration_type_words(
    header_name: &str,
    declaration: &str,
    typedefs: &BTreeMap<String, Vec<String>>,
) {
    let tokens: Vec<String> = canonical_c_tokens(declaration)
        .split(' ')
        .filter(|token| !token.is_empty())
        .map(str::to_string)
        .collect();
    let is_identifier =
        |token: &str| token.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_');
    for (index, token) in tokens.iter().enumerate() {
        if !is_identifier(token) {
            continue;
        }
        let known = NUMERIC_C_TYPES.contains(&token.as_str())
            || NON_NUMERIC_C_TYPE_WORDS.contains(&token.as_str());
        let next = tokens.get(index + 1).map_or("", String::as_str);
        if known || next == "*" || is_identifier(next) {
            assert!(
                known || typedefs.contains_key(token) || token.starts_with("chelis_"),
                "{}UNKNOWN TYPE WORD `{token}` in `{header_name}: {declaration}`: \
                 the census classifies a declaration by the arithmetic width \
                 its type words carry, so a spelling it does not recognize \
                 produces an UNFLAGGED row - which is a citation path past the \
                 rule that flagged rows have none. An allowlist of arithmetic \
                 spellings can never be complete, so the closed list is the \
                 other one. Either add `{token}` to `NUMERIC_C_TYPES` (it \
                 carries a dtype) or to `NON_NUMERIC_C_TYPE_WORDS` (it \
                 provably does not), in a change set a human reviews.{}",
                teaching_header(),
                teaching_footer()
            );
        } else {
            assert!(
                !token.starts_with('_') && !token.ends_with("_t"),
                "{}RESERVED-SHAPED DECLARATOR NAME `{token}` in \
                 `{header_name}: {declaration}`: a leading underscore is the \
                 implementation-reserved identifier namespace and a `_t` \
                 suffix is the type-alias convention, so a name of this shape \
                 means the census read a TYPE word as the thing being \
                 declared and skipped classifying it. Rename the declarator.{}",
                teaching_header(),
                teaching_footer()
            );
        }
    }
}

/// `HEADER_ROOTS` is a hand-maintained list, and §C6 forbids the census
/// from depending on one. The list survives because it also records WHY
/// each root is published, but it is no longer TRUSTED: the INCLUDE
/// closure must account for every `.h` in the published include directory,
/// so a header dropped in but reachable from no root fails loudly instead
/// of being silently absent from the inventory (round-3 red team P2).
fn assert_roots_reach_every_published_header(
    include_dir: &Path,
    source_closure: &BTreeMap<String, String>,
) {
    let on_disk = published_headers_on_disk(include_dir);
    // Reachability is a property of the raw include graph, not of whether
    // preprocessing happens to emit a locally-attributed declaration bucket.
    // A root such as `chelis_blas.h` may contribute only a system include on
    // one platform and therefore have no `per_file` output at all.
    let reached: BTreeSet<String> = source_closure.keys().cloned().collect();
    assert_eq!(
        on_disk,
        reached,
        "{}PUBLISHED HEADER NOT REACHED FROM ANY ROOT: every `.h` under {} \
         must appear in the include closure of the declared roots. A \
         header that no root includes is shipped but uninventoried, so \
         every declaration in it is invisible to this census. Add it to \
         `HEADER_ROOTS`, include it from a root, or remove it from the \
         published directory.{}",
        teaching_header(),
        include_dir.display(),
        teaching_footer()
    );
}

/// Every `.h` file the published include directory ships, keyed the way an
/// `#include` spells it: a path RELATIVE to the include directory, so
/// `sub/x.h` compares against the closure key `sub/x.h`. The walk recurses
/// (round-4 red team N5); a flat `read_dir` made a subdirectory an
/// uninventoried publishing channel, which is the same hole the
/// derived-roots rule exists to close one level up.
fn published_headers_on_disk(include_dir: &Path) -> BTreeSet<String> {
    fn walk(dir: &Path, include_dir: &Path, out: &mut BTreeSet<String>) {
        let entries = fs::read_dir(dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display()));
        for entry in entries {
            let path = entry.expect("dir entry").path();
            if path.is_dir() {
                walk(&path, include_dir, out);
            } else if path.extension().is_some_and(|ext| ext == "h") {
                out.insert(
                    path.strip_prefix(include_dir)
                        .expect("under the include directory")
                        .to_string_lossy()
                        .replace('\\', "/"),
                );
            }
        }
    }
    let mut out = BTreeSet::new();
    walk(include_dir, include_dir, &mut out);
    out
}

/// Recognize both literal include spellings. Resolution separately preserves
/// the compiler's search order: a quoted include first tries the including
/// header's directory, while an angle include uses the published include root.
/// Both spellings participate in attribution and conditional-macro guards.
fn local_include(line: &str) -> Option<&str> {
    c_include::include_name(line)
}

fn header_source_closure(include_dir: &Path, roots: &[&str]) -> BTreeMap<String, String> {
    let mut pending: Vec<String> = roots.iter().map(|root| (*root).to_string()).collect();
    let mut sources = BTreeMap::new();
    while let Some(name) = pending.pop() {
        if sources.contains_key(&name) {
            continue;
        }
        let path = include_dir.join(&name);
        let source =
            fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        for (_, line) in logical_lines(&strip_c_comments(&source)) {
            if let Some(included) = c_include::resolve(&name, &line, |candidate| {
                include_dir.join(candidate).is_file()
            }) {
                pending.push(included);
            }
        }
        sources.insert(name, source);
    }
    sources
}

/// `#line` (and a hand-written linemarker) rewrites the preprocessor's file
/// attribution, which is precisely how the census decides that a
/// declaration belongs to a published header. A declaration bracketed by
/// `#line 1 "/opt/vendor/x.h"` is still callable ABI - `cc -fsyntax-only`
/// accepts calls to it - while the census attributes it to a file outside
/// the include directory and drops it (round-3 red team P1). The directive
/// is banned in the published closure rather than interpreted: there is no
/// legitimate use of it in a hand-written published header, and any
/// interpretation would re-create the spoofing channel.
fn assert_no_line_directives(sources: &BTreeMap<String, String>) {
    for (name, source) in sources {
        for (line_number, line) in logical_lines(&strip_c_comments(source)) {
            let Some(rest) = line.trim_start().strip_prefix('#') else {
                continue;
            };
            let rest = rest.trim_start();
            let named = rest
                .strip_prefix("line")
                .is_some_and(|tail| tail.is_empty() || tail.starts_with(char::is_whitespace));
            let linemarker = rest.starts_with(|c: char| c.is_ascii_digit());
            assert!(
                !(named || linemarker),
                "{}LINE-DIRECTIVE SPOOFING SURFACE in `{name}` line {}: `{}`. \
                 A `#line` directive or hand-written linemarker rewrites the \
                 file attribution the census reads back from `cc -E`, so a \
                 real, callable export can be attributed to a file outside \
                 the published include directory and vanish from the \
                 inventory. Published headers may not contain either; delete \
                 the directive.{}",
                teaching_header(),
                line_number,
                line.trim(),
                teaching_footer()
            );
        }
    }
}

/// Physical lines joined across phase-2 line splices, each paired with the
/// physical line it starts on. A backslash-newline is deleted before any
/// directive is recognized, so `#\<newline>line 1 "/opt/x.h"` IS a `#line`
/// directive; scanning physical lines saw `#\` and a bare `line 1 "..."`
/// and let it through to the independent attribution backstop, which
/// rejected it for the right reason under the wrong name (round-4 red team,
/// diagnostic quality only - the export never entered the inventory).
fn logical_lines(text: &str) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut start = 1usize;
    for (index, line) in text.lines().enumerate() {
        if current.is_empty() {
            start = index + 1;
        }
        if let Some(head) = line.strip_suffix('\\') {
            current.push_str(head);
            continue;
        }
        current.push_str(line);
        out.push((start, std::mem::take(&mut current)));
    }
    if !current.is_empty() {
        out.push((start, current));
    }
    out
}

/// Attribution totality: every declarator the RAW published closure
/// declares must reappear in some preprocessed bucket. The linemarker ban
/// above removes the known spoofing channel; this is the independent
/// backstop that does not depend on having enumerated the channels. Names
/// are compared rather than declarations because macro expansion legally
/// rewrites type spellings between the two forms.
fn assert_total_attribution(
    sources: &BTreeMap<String, String>,
    per_file: &BTreeMap<String, String>,
) {
    let attributed: BTreeSet<String> = per_file
        .values()
        .flat_map(|text| declared_names(text))
        .collect();
    let mut missing: Vec<String> = Vec::new();
    for (name, source) in sources {
        for declared in declared_names(source) {
            if !attributed.contains(&declared) {
                missing.push(format!("{name}: {declared}"));
            }
        }
    }
    assert!(
        missing.is_empty(),
        "{}UNATTRIBUTED PUBLISHED DECLARATION {missing:?}: the raw published \
         header closure declares these names, but no preprocessed bucket \
         attributed to a file under the include directory contains them. \
         Every published declaration must land in the inventory; a \
         declaration the census cannot attribute is invisible ABI.{}",
        teaching_header(),
        teaching_footer()
    );
}

/// The declarator identities a header body declares, independent of type
/// spellings. Attribution totality is checked on these names so that
/// legitimate macro expansion (a `#define`d return type) does not read as a
/// missing declaration.
fn declared_names(text: &str) -> BTreeSet<String> {
    header_rows_local("attribution.h", text)
        .iter()
        .filter_map(|row| declarator_name(&row.kind, &row.id))
        .collect()
}

fn declarator_name(kind: &str, id: &str) -> Option<String> {
    let declaration = id.split_once(": ").map_or(id, |(_, rest)| rest);
    let tokens: Vec<&str> = declaration.split_whitespace().collect();
    let is_identifier =
        |token: &str| token.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_');
    if kind == "header-export" {
        let open = tokens.iter().position(|token| *token == "(")?;
        return tokens
            .get(open.checked_sub(1)?)
            .filter(|token| is_identifier(token))
            .map(|token| (*token).to_string());
    }
    tokens
        .iter()
        .rev()
        .find(|token| is_identifier(token))
        .map(|token| (*token).to_string())
}

fn include_guard_name(source: &str) -> Option<String> {
    let directives: Vec<&str> = source
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with('#'))
        .take(2)
        .collect();
    let guard = directives.first()?.strip_prefix("#ifndef")?.trim();
    let defined = directives.get(1)?.strip_prefix("#define")?.trim();
    (guard == defined).then(|| guard.to_string())
}

fn conditionally_defined_macros(source: &str) -> BTreeSet<String> {
    let guard = include_guard_name(source);
    let mut conditional_stack = Vec::new();
    let mut macros = BTreeSet::new();
    for line in strip_c_comments(source).lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed
            .strip_prefix("#ifdef")
            .or_else(|| trimmed.strip_prefix("#ifndef"))
            .or_else(|| trimmed.strip_prefix("#if"))
        {
            let is_guard = conditional_stack.is_empty()
                && guard.as_deref().is_some_and(|name| rest.trim() == name);
            conditional_stack.push(!is_guard);
            continue;
        }
        if trimmed.starts_with("#endif") {
            conditional_stack.pop();
            continue;
        }
        if conditional_stack.iter().any(|frame| *frame)
            && let Some(rest) = trimmed.strip_prefix("#define")
            && let Some(macro_name) = rest.split_whitespace().next()
        {
            macros.insert(
                macro_name
                    .split('(')
                    .next()
                    .expect("split always has first")
                    .to_string(),
            );
        }
    }
    macros
}

/// Conditional macro definitions taint the entire connected local-include
/// component. C preprocessing is textual: a parent definition affects a
/// child, and a child definition can affect the parent's declarations after
/// the include returns. Component-wide propagation is the conservative exact
/// policy; a public declaration may not consume any such token.
fn closure_conditional_macro_taint(
    sources: &BTreeMap<String, String>,
) -> BTreeMap<String, BTreeSet<String>> {
    let mut taint: BTreeMap<String, BTreeSet<String>> = sources
        .iter()
        .map(|(name, source)| (name.clone(), conditionally_defined_macros(source)))
        .collect();
    let edges: Vec<(String, String)> = sources
        .iter()
        .flat_map(|(name, source)| {
            logical_lines(&strip_c_comments(source))
                .into_iter()
                .filter_map(|(_, line)| {
                    c_include::resolve(name, &line, |candidate| sources.contains_key(candidate))
                })
                .map(|included| (name.clone(), included))
                .collect::<Vec<_>>()
        })
        .collect();
    loop {
        let mut changed = false;
        for (left, right) in &edges {
            let component: BTreeSet<String> = taint[left].union(&taint[right]).cloned().collect();
            let left_set = taint.get_mut(left).expect("edge endpoint exists");
            let old_left = left_set.len();
            left_set.extend(component.iter().cloned());
            changed |= left_set.len() != old_left;
            let right_set = taint.get_mut(right).expect("edge endpoint exists");
            let old_right = right_set.len();
            right_set.extend(component.iter().cloned());
            changed |= right_set.len() != old_right;
        }
        if !changed {
            break;
        }
    }
    taint
}

/// The supported preprocessing policy is total by construction: public ABI
/// declarations in the published local-header closure may not be conditional.
/// Platform and feature branches remain permitted inside `static` function
/// bodies and for include/macro selection that does not declare ABI.
fn assert_context_invariant_headers(include_dir: &Path, roots: &[&str]) {
    let sources = header_source_closure(include_dir, roots);
    let closure_macro_taint = closure_conditional_macro_taint(&sources);
    for (name, source) in &sources {
        let guard = include_guard_name(source);
        let mut conditional_stack: Vec<bool> = Vec::new();
        let mut brace_depth = 0usize;
        let mut conditional_top_level = String::new();
        let mut conditional_includes = BTreeSet::new();

        for line in strip_c_comments(source).lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed
                .strip_prefix("#ifdef")
                .or_else(|| trimmed.strip_prefix("#ifndef"))
                .or_else(|| trimmed.strip_prefix("#if"))
            {
                let is_guard = conditional_stack.is_empty()
                    && guard.as_deref().is_some_and(|g| rest.trim() == g);
                conditional_stack.push(!is_guard);
                continue;
            }
            if trimmed.starts_with("#elif") || trimmed == "#else" {
                continue;
            }
            if trimmed.starts_with("#endif") {
                conditional_stack.pop();
                continue;
            }

            let varying = conditional_stack.iter().any(|frame| *frame);
            if varying
                && let Some(included) =
                    c_include::resolve(name, trimmed, |candidate| sources.contains_key(candidate))
            {
                conditional_includes.insert(included);
            }
            let extern_wrapper = trimmed == "extern \"C\" {" || trimmed == "}";
            if varying && brace_depth == 0 && !trimmed.starts_with('#') {
                conditional_top_level.push_str(line);
                conditional_top_level.push('\n');
            }
            if !extern_wrapper {
                for c in line.chars() {
                    if c == '{' {
                        brace_depth += 1;
                    } else if c == '}' {
                        brace_depth = brace_depth.saturating_sub(1);
                    }
                }
            }
        }

        let conditional_rows = header_rows_local(name, &conditional_top_level);
        let conditional_typedef = conditional_top_level
            .split(';')
            .any(|statement| normalize_ws(statement).starts_with("typedef "));
        let raw_rows = header_rows_local(name, source);
        let tainted_macros = &closure_macro_taint[name];
        let mut macro_taint_hits = BTreeSet::new();
        let macro_dependent_rows: Vec<&Row> = raw_rows
            .iter()
            .filter(|row| {
                let tokens: BTreeSet<&str> = row
                    .id
                    .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                    .filter(|token| !token.is_empty())
                    .collect();
                let hits: Vec<String> = tainted_macros
                    .iter()
                    .filter(|macro_name| tokens.contains(macro_name.as_str()))
                    .cloned()
                    .collect();
                let dependent = !hits.is_empty();
                macro_taint_hits.extend(hits);
                dependent
            })
            .collect();
        // A conditional include can expose ABI through an otherwise empty
        // intermediate header. Follow its complete local closure before
        // deciding that the conditional branch declares no public surface.
        let mut pending: Vec<_> = conditional_includes.iter().cloned().collect();
        while let Some(included) = pending.pop() {
            for (_, line) in logical_lines(&strip_c_comments(&sources[&included])) {
                if let Some(child) = c_include::resolve(&included, &line, |candidate| {
                    sources.contains_key(candidate)
                }) && conditional_includes.insert(child.clone())
                {
                    pending.push(child);
                }
            }
        }
        let conditional_include_rows: Vec<Row> = conditional_includes
            .iter()
            .filter_map(|included| sources.get(included).map(|source| (included, source)))
            .flat_map(|(included, source)| header_rows_local(included, source))
            .collect();
        assert!(
            conditional_rows.is_empty()
                && !conditional_typedef
                && macro_dependent_rows.is_empty()
                && conditional_include_rows.is_empty(),
            "{}CONTEXT-VARYING PUBLIC ABI in `{name}`: conditional branches \
             contain exported declarations {:?}, a top-level typedef is \
             conditional ({conditional_typedef}), conditional macros reach \
             declarations {:?} via tainted tokens {:?}, or conditional local \
             includes expose {:?}. \
             Published ABI declarations and their type spellings must be \
             unconditional in the local header closure; conditional code is \
             permitted only behind static implementation details.{}",
            teaching_header(),
            conditional_rows
                .iter()
                .map(|row| row.id.as_str())
                .collect::<Vec<_>>(),
            macro_dependent_rows
                .iter()
                .map(|row| row.id.as_str())
                .collect::<Vec<_>>(),
            macro_taint_hits,
            conditional_include_rows
                .iter()
                .map(|row| row.id.as_str())
                .collect::<Vec<_>>(),
            teaching_footer()
        );
    }
}

fn strip_c_comments(text: &str) -> String {
    c_lexical::strip_c_comments(text)
}

fn normalize_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Tokenize the declaration subset of C used by published headers. Identity
/// is the token sequence, never a preprocessor's incidental whitespace.
fn canonical_c_tokens(s: &str) -> String {
    c_lexical::canonical_c_tokens(s)
}

/// Fixed-width numeric C value types: a signature mentioning one (after
/// typedef resolution) is a numeric runtime callable and carries the
/// `numeric-op` flag, which binds rows to semantic registration. Every
/// non-boolean, non-character built-in arithmetic spelling is conservative
/// numeric capacity; there is no integer-plumbing exemption.
const NUMERIC_C_TYPES: &[&str] = c_lexical::NUMERIC_C_TYPES;

/// The FROZEN set of type words a published declaration may use that carry
/// no arithmetic width: storage/qualifier noise, the aggregate keywords, and
/// the two non-arithmetic value spellings. It exists so the classification
/// rule can be INVERTED (`assert_known_type_words`): membership in
/// `NUMERIC_C_TYPES` decides the flags, but a type word in NEITHER list is
/// a loud rejection rather than an unflagged row.
///
/// Round-4 red team N1 is why. `NUMERIC_C_TYPES` alone is an allowlist, so
/// `_Float16`, `__fp16`, `__bf16`, `_Decimal64`, and `__int128` classified
/// as `[]` - a bare-float export could enter the census on an ordinary
/// issue citation, past the seam rule that has no citation path. An
/// allowlist of arithmetic spellings can never be complete (every
/// toolchain adds its own), so the closed set has to be the OTHER one: the
/// words that are known not to carry a dtype. A new arithmetic spelling
/// then arrives as a build failure naming the unknown word, which is the
/// same footing an unresolvable typedef already has.
const NON_NUMERIC_C_TYPE_WORDS: &[&str] = c_lexical::NON_NUMERIC_C_TYPE_WORDS;

/// The flags that make a row an untagged capacity seam. `numeric-op` is
/// classification, not a seam.
fn is_seam(flags: &[String]) -> bool {
    flags
        .iter()
        .any(|f| f == "float-carrier" || f == "raw-dtype-int")
}

/// Collect `typedef` aliases so classification sees through spellings like
/// `typedef int chelis_dtype_id;` - the re-red-team's executed typedef
/// evasion. Struct forward typedefs resolve to their `struct X` spelling,
/// which is harmless. Aggregate bodies (`{ ... }`) are inventoried as
/// `header-struct` rows instead. A FUNCTION-POINTER typedef resolves to its
/// full return/parameter word list, so a setter taking the callback
/// inherits the callback's numeric and dtype words (round-3 red team P1: a
/// skipped `typedef double (*cb)(double, int elem_dtype);` launders a seam
/// into an empty-flag row). Any other parenthesized typedef is REJECTED
/// rather than skipped, so the resolution path stays total.
fn collect_typedefs(text: &str) -> BTreeMap<String, Vec<String>> {
    let mut map = BTreeMap::new();
    for stmt in text.split(';') {
        let stmt = normalize_ws(stmt);
        let Some(rest) = stmt.strip_prefix("typedef ") else {
            continue;
        };
        if rest.contains('{') {
            continue;
        }
        if rest.contains('(') {
            let (name, target) = function_pointer_typedef(&stmt).unwrap_or_else(|| {
                panic!(
                    "{}UNRESOLVABLE PARENTHESIZED TYPEDEF `{stmt};`: the census \
                     resolves typedef spellings before classifying a \
                     declaration, and a typedef it cannot resolve hides every \
                     numeric and dtype word behind an opaque name. Only the \
                     `typedef <return...> (*<name>)(<params...>)` \
                     function-pointer shape is supported; rewrite the \
                     declaration or teach `function_pointer_typedef` the new \
                     shape in the same change set.{}",
                    teaching_header(),
                    teaching_footer()
                )
            });
            map.insert(name, target);
            continue;
        }
        if rest.contains('[') {
            let (name, target) = array_typedef(&stmt).unwrap_or_else(|| {
                panic!(
                    "{}UNRESOLVABLE ARRAY TYPEDEF `{stmt};`: an array typedef \
                     declares its alias immediately before the first `[`, and \
                     one this resolver cannot read that way hides every \
                     numeric and dtype word behind an opaque name. The generic \
                     word split reads the array EXTENT as the alias, so a \
                     setter taking the alias inherits nothing and a \
                     `float-carrier` seam becomes an unflagged row (round-4 \
                     red team N2). Rewrite the declaration or teach \
                     `array_typedef` the new shape in the same change set.{}",
                    teaching_header(),
                    teaching_footer()
                )
            });
            map.insert(name, target);
            continue;
        }
        let mut words: Vec<String> = rest
            .split(|c: char| !(c.is_alphanumeric() || c == '_'))
            .filter(|w| !w.is_empty())
            .map(|w| w.to_string())
            .collect();
        if words.len() >= 2 {
            let name = words.pop().expect("nonempty");
            map.insert(name, words);
        }
    }
    map
}

/// Split `typedef <element...> <name>[<extent>]...` into the alias name and
/// the element type's words. The alias is the identifier immediately before
/// the FIRST `[`; everything between `typedef` and it is the element type,
/// so `typedef double chelis_vec4[4];` makes a setter taking `chelis_vec4`
/// inherit `double` (round-4 red team N2 - the generic word split popped
/// the array extent as the alias and registered `4`, leaving the real alias
/// unresolved and its `float-carrier` seam unflagged). A shape that does
/// not read that way, including an empty element type or an alias that is
/// itself a type keyword, is rejected rather than guessed at.
fn array_typedef(stmt: &str) -> Option<(String, Vec<String>)> {
    let tokens: Vec<String> = canonical_c_tokens(stmt)
        .split(' ')
        .map(str::to_string)
        .collect();
    let open = tokens.iter().position(|token| token == "[")?;
    let name = tokens.get(open.checked_sub(1)?)?;
    if !name.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
        || NUMERIC_C_TYPES.contains(&name.as_str())
        || NON_NUMERIC_C_TYPE_WORDS.contains(&name.as_str())
    {
        return None;
    }
    let target: Vec<String> = tokens[..open - 1]
        .iter()
        .filter(|token| {
            *token != "typedef"
                && token.starts_with(|c: char| c.is_ascii_alphanumeric() || c == '_')
        })
        .cloned()
        .collect();
    if target.is_empty() {
        return None;
    }
    Some((name.clone(), target))
}

/// Split `typedef <return...> (*<name>)(<params...>)` into the alias name
/// and every other word of its signature, so `resolve_words` expands a
/// callback parameter into the return and parameter spellings it carries.
fn function_pointer_typedef(stmt: &str) -> Option<(String, Vec<String>)> {
    let tokens: Vec<String> = canonical_c_tokens(stmt)
        .split(' ')
        .map(str::to_string)
        .collect();
    let is_identifier =
        |token: &String| token.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_');
    let anchor = tokens.windows(4).position(|window| {
        window[0] == "(" && window[1] == "*" && is_identifier(&window[2]) && window[3] == ")"
    })?;
    let name = tokens[anchor + 2].clone();
    let target = tokens
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != anchor + 2)
        .map(|(_, token)| token.clone())
        .filter(|token| {
            token != "typedef" && token.starts_with(|c: char| c.is_ascii_alphanumeric() || c == '_')
        })
        .collect();
    Some((name, target))
}

/// Classification shapes the enforcement rule a row falls under; the
/// citation requirement applies to EVERY inventory change, so renaming a
/// parameter to dodge a flag dodges nothing, and typedef/macro spellings
/// are resolved before classifying (transitively, depth-capped, inside
/// `c_lexical::resolve_words`).
fn classify(sig: &str, typedefs: &BTreeMap<String, Vec<String>>) -> Vec<String> {
    c_lexical::classify(sig, typedefs)
}

/// Lock the shared lexical primitives against the exact spellings the census
/// depends on. These moved out of the tripwire body into
/// `tests/support/c_lexical.rs` so every census leg shares one implementation;
/// the composition through THIS file's `collect_typedefs` is the real consumer
/// path, so that is what this pins.
#[test]
fn shared_capacity_census_primitives_keep_their_contract() {
    assert_eq!(
        c_lexical::canonical_c_tokens("const  bool * value;"),
        "const _Bool * value ;"
    );
    assert_eq!(
        c_lexical::strip_c_comments("float /* x */ *p; // y\n"),
        "float   *p; \n"
    );
    let typedefs = collect_typedefs("typedef double sample; typedef sample alias;");
    assert_eq!(
        classify("alias *value", &typedefs),
        vec!["float-carrier", "numeric-op"]
    );
}

/// A comment or literal must never hide a declaration from the census, and a
/// declarator inside a string must never enter it. This is the behavior that
/// widened when the stripper moved: the previous inline copy was byte-based
/// and had no literal awareness.
#[test]
fn comment_stripping_respects_string_and_character_literals() {
    assert_eq!(
        c_lexical::strip_c_comments(r#"const char *s = "a /* not a comment */ b";"#),
        r#"const char *s = "a /* not a comment */ b";"#
    );
    assert_eq!(
        c_lexical::strip_c_comments("char c = '/'; // tail\n"),
        "char c = '/'; \n"
    );
}

/// Extract exported declarations and struct layouts from one preprocessed
/// header body. `static` definitions carry no ABI and are skipped; the
/// `extern "C" {` wrapper is neutralized; preprocessor lines are dropped.
/// Planted-test convenience: classify with the typedefs found in the
/// same text (the real pipeline builds a global map across headers).
fn header_rows_local(header_name: &str, raw: &str) -> Vec<Row> {
    let typedefs = collect_typedefs(&strip_c_comments(raw));
    header_rows(header_name, raw, &typedefs)
}

/// The declaration subset that names a function: a parameter list is the
/// only thing separating a callable from published data.
fn is_callable(declaration: &str) -> bool {
    declaration.contains('(') && declaration.ends_with(')')
}

fn push_callable_row(
    rows: &mut Vec<Row>,
    header_name: &str,
    declaration: &str,
    typedefs: &BTreeMap<String, Vec<String>>,
) {
    let flags = classify(declaration, typedefs);
    let id = format!(
        "{header_name}: {}",
        canonical_c_tokens(&format!("{declaration};"))
    );
    rows.push(Row {
        kind: "header-export".to_string(),
        id,
        flags,
        citation: String::new(),
    });
}

fn header_rows(header_name: &str, raw: &str, typedefs: &BTreeMap<String, Vec<String>>) -> Vec<Row> {
    let text = strip_c_comments(raw);
    let text: String = logical_lines(&text)
        .into_iter()
        .filter(|(_, line)| !line.trim_start().starts_with('#'))
        .map(|(_, line)| line)
        .collect::<Vec<_>>()
        .join("\n")
        .replace("extern \"C\" {", "");

    let mut rows = Vec::new();
    let mut seg = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '{' => {
                let head = normalize_ws(&seg);
                // Consume the brace-matched body in every case; what differs
                // is whether the construct publishes ABI.
                let mut depth = 1usize;
                let mut body = String::new();
                for c2 in chars.by_ref() {
                    match c2 {
                        '{' => depth += 1,
                        '}' => {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        _ => {}
                    }
                    body.push(c2);
                }
                if head.starts_with("static") {
                    // A `static inline` definition carries no ABI export.
                } else if head.starts_with("typedef") || (!head.is_empty() && !is_callable(&head)) {
                    // A published type layout: `typedef struct/enum/union
                    // { ... } name;` and its untypedef'd forms. Capture the
                    // trailing declarator too - without it the `enum` tail
                    // would fall through to the statement arm as a bare name.
                    let mut tail = String::new();
                    for c2 in chars.by_ref() {
                        if c2 == ';' {
                            break;
                        }
                        tail.push(c2);
                    }
                    let declaration = format!(
                        "{} {{ {} }} {}",
                        head,
                        normalize_ws(&body),
                        normalize_ws(&tail)
                    );
                    let flags = classify(&declaration, typedefs);
                    rows.push(Row {
                        kind: "header-struct".to_string(),
                        id: format!("{header_name}: {}", canonical_c_tokens(&declaration)),
                        flags,
                        citation: String::new(),
                    });
                } else if is_callable(&head) {
                    // A non-static function DEFINITION in a published header
                    // is an external definition, so it is ABI exactly as its
                    // declaration would be.
                    push_callable_row(&mut rows, header_name, &head, typedefs);
                }
                seg.clear();
            }
            '}' => {
                // Orphaned closer from the neutralized extern "C" block.
                seg.clear();
            }
            ';' => {
                let stmt = normalize_ws(&seg);
                seg.clear();
                if stmt.is_empty() || stmt.starts_with("typedef") || stmt.starts_with("static") {
                    continue;
                }
                if is_callable(&stmt) {
                    push_callable_row(&mut rows, header_name, &stmt, typedefs);
                } else {
                    // Non-function published ABI: `extern double
                    // chelis_global_scale;` is a numeric channel with no
                    // callable to classify (round-3 red team P1). Inventorying
                    // every leftover statement keeps the statement arm total,
                    // so a future non-function declaration form cannot be
                    // silently dropped.
                    rows.push(Row {
                        kind: "header-data".to_string(),
                        id: format!("{header_name}: {}", canonical_c_tokens(&format!("{stmt};"))),
                        flags: classify(&stmt, typedefs),
                        citation: String::new(),
                    });
                }
            }
            _ => seg.push(c),
        }
    }
    rows
}

#[test]
fn multiline_preprocessor_macros_do_not_create_phantom_exports() {
    let rows = header_rows_local(
        "fixture.h",
        concat!(
            "#define CHECK(call) do { \\\n",
            "    if ((call) != 0) abort(); \\\n",
            "} while (0)\n",
            "int real_export(void);\n",
        ),
    );
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert!(rows[0].id.contains("real_export"));
}

// ---------------------------------------------------------------------------
// Leg B: numeric ADT variants in the desugared stdlib AST
// ---------------------------------------------------------------------------

fn walk_ch_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            walk_ch_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "ch") {
            out.push(path);
        }
    }
}

/// Retained capacity flags for concrete numeric payloads. Final authority
/// classifies a declared Chelis primitive as a typed numeric operation; these
/// flags do not turn a language ADT into an untagged external C carrier.
fn numeric_carrier_flags(prims: &BTreeSet<String>) -> Vec<String> {
    let mut flags = Vec::new();
    if prims.iter().any(|p| FLOAT_PRIMS.contains(&p.as_str())) {
        flags.push("float-carrier".to_string());
    }
    if prims.iter().any(|p| !FLOAT_PRIMS.contains(&p.as_str())) {
        flags.push("numeric-op".to_string());
    }
    flags
}

#[path = "../../../tests/support/capacity_census_stdlib.rs"]
mod stdlib_discovery;
use stdlib_discovery::{scan_deftypes, stdlib_rows};

// ---------------------------------------------------------------------------
// Rust-registered prelude value ADTs (the chelis#890 `Json` shape)
// ---------------------------------------------------------------------------

/// Deterministic census rendering of one prelude ADT field type. Not the
/// Deep printer (a Rust-registered `chelis_types::Type` never passes
/// through Deep), but the same identity discipline: type name, variant
/// list, and per-field type spelling, so any change to a numeric field's
/// position or dtype changes the row id and fails the diff.
fn render_prelude_census_type(ty: &chelis_types::types::Type) -> String {
    use chelis_types::types::Type;
    match ty {
        Type::Prim(p) => p.name().to_string(),
        Type::Adt(name, args) => {
            if args.is_empty() {
                name.clone()
            } else {
                format!(
                    "{name}[{}]",
                    args.iter()
                        .map(render_prelude_census_type)
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
        }
        Type::Tuple(items) => format!(
            "({})",
            items
                .iter()
                .map(render_prelude_census_type)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Type::Ref(inner) => format!("&{}", render_prelude_census_type(inner)),
        Type::Fn(args, ret) => format!(
            "({}) -> {}",
            args.iter()
                .map(render_prelude_census_type)
                .collect::<Vec<_>>()
                .join(", "),
            render_prelude_census_type(ret)
        ),
        Type::Var(v) => format!("t{}", v.0),
        Type::Unit => "unit".to_string(),
        // A tensor or error type inside a prelude ADT registration would
        // itself be new surface; render it loudly rather than skipping.
        other => format!("<unrenderable {other:?}>"),
    }
}

/// Collect the numeric primitive spellings reachable in one prelude ADT
/// field type, the resolved stdlib traversal counterpart for
/// Rust-registered types.
fn collect_prelude_numeric_prims(ty: &chelis_types::types::Type, prims: &mut BTreeSet<String>) {
    use chelis_types::types::{TensorPrec, Type};
    match ty {
        Type::Prim(p) => {
            if NUMERIC_PRIMS.contains(&p.name()) {
                prims.insert(p.name().to_string());
            }
        }
        Type::Adt(_, args) | Type::Tuple(args) | Type::Fn(args, _) => {
            for arg in args {
                collect_prelude_numeric_prims(arg, prims);
            }
            if let Type::Fn(_, ret) = ty {
                collect_prelude_numeric_prims(ret, prims);
            }
        }
        Type::Ref(inner) => collect_prelude_numeric_prims(inner, prims),
        Type::Tensor(_, TensorPrec::Concrete(p)) if NUMERIC_PRIMS.contains(&p.name()) => {
            prims.insert(p.name().to_string());
        }
        _ => {}
    }
}

/// The prelude-adt-numeric leg (spec/design/dtype_semantics.md §C6): the
/// Rust-registered prelude value ADTs, enumerated from the single
/// registration path (`chelis_types::prelude_adt_defs`, which rebuilds
/// through `register_prelude_adts`). The final #1293 surface intentionally
/// has no numeric prelude ADT, but this leg stays executable so adding one
/// cannot create an unenumerated numeric channel. Same classification rule
/// as every other family: a float primitive is a `float-carrier` seam, an
/// integer primitive is `numeric-op`.
fn prelude_adt_rows() -> Vec<Row> {
    let mut rows = Vec::new();
    for def in chelis_types::prelude_adt_defs() {
        let mut prims = BTreeSet::new();
        for variant in &def.variants {
            for (_, ty) in &variant.fields {
                collect_prelude_numeric_prims(ty, &mut prims);
            }
        }
        if prims.is_empty() {
            continue;
        }
        let shape = def
            .variants
            .iter()
            .map(|variant| {
                if variant.fields.is_empty() {
                    variant.name.clone()
                } else {
                    format!(
                        "{}({})",
                        variant.name,
                        variant
                            .fields
                            .iter()
                            .map(|(_, ty)| render_prelude_census_type(ty))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
            })
            .collect::<Vec<_>>()
            .join(" | ");
        rows.push(Row {
            kind: "prelude-adt-numeric".to_string(),
            id: format!("prelude::{}: {}", def.name, shape),
            flags: numeric_carrier_flags(&prims),
            citation: String::new(),
        });
    }
    rows
}

// ---------------------------------------------------------------------------
// The inventory, the baseline, and the diff
// ---------------------------------------------------------------------------

fn current_inventory(root: &Path) -> Vec<Row> {
    let include_dir = root.join(INCLUDE_DIR_REL);
    let mut rows = Vec::new();
    let per_file = preprocessed_headers(&include_dir, HEADER_ROOTS);
    let mut typedefs = BTreeMap::new();
    for text in per_file.values() {
        typedefs.append(&mut collect_typedefs(text));
    }
    for (name, text) in &per_file {
        rows.extend(header_rows(name, text, &typedefs));
    }
    rows.extend(stdlib_rows(root));
    rows.extend(prelude_adt_rows());
    rows.sort_by(|a, b| (a.kind.as_str(), a.id.as_str()).cmp(&(b.kind.as_str(), b.id.as_str())));
    rows.dedup_by(|a, b| a.kind == b.kind && a.id == b.id);
    rows
}

fn teaching_header() -> String {
    "capacity census violation \
     (spec/design/dtype_semantics.md §C6 deliverable 1; \
     AGENTS.md §Numeric Surface Discipline; chelis#729)\n"
        .to_string()
}

fn teaching_footer() -> String {
    "\nSanctioned actions:\n\
     1. STRUCTURALLY NONNUMERIC: add the exact family/kind/canonical-id/flags \
     descriptor to the nonnumeric registry only after proving that it carries no \
     numeric capacity. Empty flags alone are not authority.\n\
     2. TAGGED TRANSPORT: redesign onto the exact tagged carrier and register the \
     complete descriptor. Resemblance, names, prefixes, and bare numeric carriers \
     are not tagged-transport authority.\n\
     3. NUMERIC OPERATION: author or select the governing exact [05-OP-N] atom in \
     spec/05 and add the family-qualified complete `NumericOperationRegistration`. \
     Re-check the highest allocated atom number on current \
     main, run .venv/bin/python scripts/generate_rejection_registries.py --write, \
     and commit crates/chelis-types/src/rejection_registry_generated.rs. The \
     generated registry is required but is not semantic authority; an unrelated or \
     nonexistent atom still fails.\n\
     4. An identity change is removal plus addition: remove the old descriptor, \
     then register the successor under exactly one of rules \
     1-3. No transition exception can be copied or newly authored. A removed ABI \
     row is 0.19 payload by \
     default per remediation_roadmap.md anti-churn invariant 7.\n\
     This test and spec/design/capacity_census.json are guard artifacts; \
     editing either to make a change pass is never the fix.\n"
        .to_string()
}

fn check_against_baseline(current: &[Row], baseline: &Baseline) -> Result<(), String> {
    let spec = fs::read_to_string(repo_root().join(CONTROLLING_SPEC_REL))
        .expect("controlling spec/05 must be readable");
    check_against_baseline_with(current, baseline, SEMANTIC_REGISTRATIONS, &spec)
}

fn callable_identity(row: &Row) -> String {
    format!("[{}] {}", row.kind, row.id)
}

fn registration_problem(registration: SemanticRegistration, spec: &str) -> Option<String> {
    let Some(atom) = registration
        .atom
        .strip_prefix('[')
        .and_then(|atom| atom.strip_suffix(']'))
    else {
        return Some(format!(
            "malformed atom `{}` (expected `[05-OP-N]`)",
            registration.atom
        ));
    };
    let parts: Vec<&str> = atom.split('-').collect();
    if parts.len() != 3
        || parts[0] != "05"
        || parts[1] != "OP"
        || parts[2].is_empty()
        || !parts[2].chars().all(|c| c.is_ascii_digit())
    {
        return Some(format!(
            "wrong atom grammar/group `{}` (expected `[05-OP-N]`)",
            registration.atom
        ));
    }
    let definition_prefix = format!("> **{}**", registration.atom);
    if !spec
        .lines()
        .any(|line| line.trim_start().starts_with(&definition_prefix))
    {
        return Some(format!(
            "atom `{}` does not exist as a normative `> **[05-OP-N]**` \
             definition in {}",
            registration.atom, CONTROLLING_SPEC_REL
        ));
    }
    None
}

fn check_against_baseline_with(
    current: &[Row],
    baseline: &Baseline,
    registrations: &[SemanticRegistration],
    spec: &str,
) -> Result<(), String> {
    check_against_baseline_with_authorities(
        current,
        baseline,
        registrations,
        spec,
        final_authority_registries(),
    )
}

fn check_against_baseline_with_authorities(
    current: &[Row],
    baseline: &Baseline,
    registrations: &[SemanticRegistration],
    spec: &str,
    final_registries: AuthorityRegistries<'_>,
) -> Result<(), String> {
    let mut problems = Vec::new();
    let mut seen_baseline = BTreeSet::new();
    for row in &baseline.rows {
        if !seen_baseline.insert((row.kind.clone(), row.id.clone())) {
            problems.push(format!(
                "DUPLICATE BASELINE DESCRIPTOR: [{}] {}",
                row.kind, row.id
            ));
        }
    }
    let mut seen_current = BTreeSet::new();
    for row in current {
        if !seen_current.insert((row.kind.clone(), row.id.clone())) {
            problems.push(format!(
                "DUPLICATE CURRENT DESCRIPTOR: [{}] {}",
                row.kind, row.id
            ));
        }
    }
    let base_map: BTreeMap<(String, String), &Row> = baseline
        .rows
        .iter()
        .map(|r| ((r.kind.clone(), r.id.clone()), r))
        .collect();
    let cur_keys: BTreeSet<(String, String)> = current
        .iter()
        .map(|r| (r.kind.clone(), r.id.clone()))
        .collect();

    let expected_manifest = coverage_manifest();
    if baseline.version != 3 || baseline.legs != expected_manifest {
        problems.push(format!(
            "INVALID COVERAGE MANIFEST: baseline version/legs do not equal \
             the fixed executable `coverage_manifest`; a leg cannot become \
             covered without a live enumerator, command, expected success, \
             and mutation_oracle. expected={expected_manifest:?}, \
             actual(version={}, legs={:?})",
            baseline.version, baseline.legs
        ));
    }

    let mut registration_map = BTreeMap::new();
    for registration in registrations {
        if registration_map
            .insert(registration.callable, *registration)
            .is_some()
        {
            problems.push(format!(
                "DUPLICATE SEMANTIC REGISTRATION for `{}`",
                registration.callable
            ));
        }
        if let Some(problem) = registration_problem(*registration, spec) {
            problems.push(format!(
                "INVALID SEMANTIC REGISTRATION for `{}`: {problem}",
                registration.callable
            ));
        }
    }

    for row in current {
        if let Some(baseline_row) = base_map.get(&(row.kind.clone(), row.id.clone())) {
            if row.flags != baseline_row.flags {
                problems.push(format!(
                    "ENFORCEMENT METADATA CHANGED for matched row [{}] {}: \
                     baseline flags {:?}, current flags {:?}",
                    row.kind, row.id, baseline_row.flags, row.flags
                ));
            }
        } else {
            problems.push(format!(
                "NEW surface not in the census: [{}] {} (flags: {:?})",
                row.kind, row.id, row.flags
            ));
        }
    }
    for row in &baseline.rows {
        if !cur_keys.contains(&(row.kind.clone(), row.id.clone())) {
            problems.push(format!(
                "REMOVED surface still in the census: [{}] {}",
                row.kind, row.id
            ));
        }
        match capacity_census_authority::classify_final_authority(
            &authority_surface(row),
            final_registries,
            spec,
        ) {
            Ok(_) if !row.citation.trim().is_empty() => problems.push(format!(
                "FINAL AUTHORITY ROW CARRIES A TRANSITION DISPOSITION: [{}] {} has `{}`; final rows omit legacy citations/exceptions",
                row.kind, row.id, row.citation
            )),
            Ok(_) => {}
            Err(problem) => problems.push(format!(
                "UNCLASSIFIED census row (no final authority; citations and transition exceptions are not authority): [{}] {} ({problem})",
                row.kind, row.id
            )),
        }
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{}{}\n{}",
            teaching_header(),
            problems.join("\n"),
            teaching_footer()
        ))
    }
}

fn regenerate(baseline_path: &Path, current: &[Row], _old: Option<&Baseline>) {
    let spec = fs::read_to_string(repo_root().join(CONTROLLING_SPEC_REL))
        .expect("controlling spec/05 must be readable");
    let rows: Vec<Row> = current
        .iter()
        .map(|r| Row {
            kind: r.kind.clone(),
            id: r.id.clone(),
            flags: r.flags.clone(),
            citation: if capacity_census_authority::classify_final_authority(
                &authority_surface(r),
                final_authority_registries(),
                &spec,
            )
            .is_ok()
            {
                String::new()
            } else {
                "TODO".to_string()
            },
        })
        .collect();
    let out = Baseline {
        version: 3,
        legs: coverage_manifest(),
        rows,
    };
    let json = serde_json::to_string_pretty(&out).expect("serialize census");
    fs::write(baseline_path, json + "\n")
        .unwrap_or_else(|e| panic!("write {}: {e}", baseline_path.display()));
}

// ---------------------------------------------------------------------------
// The tripwire
// ---------------------------------------------------------------------------

#[test]
fn capacity_census_matches_public_surface() {
    let root = repo_root();
    let baseline_path = root.join(BASELINE_REL);
    let current = current_inventory(&root);
    assert!(
        !current.is_empty(),
        "capacity census enumerated nothing; the enumerators are broken, \
         which would make every future addition invisible"
    );

    let old: Option<Baseline> = fs::read_to_string(&baseline_path)
        .ok()
        .map(|text| serde_json::from_str(&text).expect("parse capacity_census.json"));

    if std::env::var("CHELIS_CAPACITY_CENSUS_WRITE").as_deref() == Ok("1") {
        regenerate(&baseline_path, &current, old.as_ref());
    }

    let baseline: Baseline =
        serde_json::from_str(&fs::read_to_string(&baseline_path).unwrap_or_else(|e| {
            panic!(
                "{}missing baseline {}: {e}\nRun once with \
                 CHELIS_CAPACITY_CENSUS_WRITE=1 to create it, then classify every \
                 TODO row through exactly one final authority.{}",
                teaching_header(),
                baseline_path.display(),
                teaching_footer()
            )
        }))
        .expect("parse capacity_census.json");

    if let Err(msg) = check_against_baseline(&current, &baseline) {
        panic!("{msg}");
    }
}

#[test]
fn tensor_entry_borrow_successor_has_exact_op44_authority() {
    const ID: &str = "chelis_runtime.h: chelis_tensor * chelis_tensor_entry_borrow ( int32_t rank , const int64_t * shape , chelis_dtype dtype , const void * data , int64_t byte_capacity ) ;";
    const SIGNATURE: &str = "`chelis_tensor *chelis_tensor_entry_borrow(int32_t rank, const int64_t *shape, chelis_dtype dtype, const void *data, int64_t byte_capacity)`";

    let root = repo_root();
    let current = current_inventory(&root);
    assert!(
        current
            .iter()
            .all(|row| !row.id.contains("chelis_alloc_view")),
        "the retired foreign-storage alias must not remain in the published census"
    );
    let row = current
        .iter()
        .find(|row| row.kind == "header-export" && row.id == ID)
        .expect("the exact entry-borrow successor is present in the published header census");
    let registration = FINAL_NUMERIC_OPERATION_ROWS
        .iter()
        .find(|registration| registration.surface.kind == row.kind && registration.surface.id == ID)
        .expect("entry borrow has an exact semantic registration");
    assert_eq!(registration.atom, "[05-OP-44]");
    assert_eq!(registration.surface.flags, &["numeric-op"]);
    assert!(
        include_str!("../../../spec/registry/c_heap_lifetime.md").contains(SIGNATURE),
        "[05-OP-44]'s incorporated registry must contain the exact successor signature"
    );
    let spec = fs::read_to_string(root.join(CONTROLLING_SPEC_REL)).expect("read spec/05");
    assert_eq!(
        capacity_census_authority::classify_final_authority(
            &authority_surface(row),
            final_authority_registries(),
            &spec,
        ),
        Ok(capacity_census_authority::FinalAuthority::NumericOperation { atom: "[05-OP-44]" })
    );
}

#[test]
fn tensor_repurpose_has_exact_op44_authority() {
    const ID: &str = "chelis_runtime.h: void chelis_tensor_repurpose ( chelis_tensor * tensor , chelis_scalar rank , const chelis_scalar * shape ) ;";
    const SIGNATURE: &str = "`void chelis_tensor_repurpose(chelis_tensor *tensor, chelis_scalar rank, const chelis_scalar *shape)`";

    let root = repo_root();
    let current = current_inventory(&root);
    let row = current
        .iter()
        .find(|row| row.kind == "header-export" && row.id == ID)
        .expect("the exact tagged repurpose callable is present in the public census");
    let registration = FINAL_NUMERIC_OPERATION_ROWS
        .iter()
        .find(|registration| registration.surface.kind == row.kind && registration.surface.id == ID)
        .expect("tensor repurpose has an exact semantic registration");
    assert_eq!(registration.atom, "[05-OP-44]");
    assert!(registration.surface.flags.is_empty());
    assert!(
        include_str!("../../../spec/registry/c_heap_lifetime.md").contains(SIGNATURE),
        "[05-OP-44]'s incorporated registry must contain the exact tagged signature"
    );
    let spec = fs::read_to_string(root.join(CONTROLLING_SPEC_REL)).expect("read spec/05");
    assert_eq!(
        capacity_census_authority::classify_final_authority(
            &authority_surface(row),
            final_authority_registries(),
            &spec,
        ),
        Ok(capacity_census_authority::FinalAuthority::NumericOperation { atom: "[05-OP-44]" })
    );
}

#[test]
fn migrated_primary_rows_have_exact_final_authority_and_no_transition_disposition() {
    let baseline: Baseline = serde_json::from_str(
        &fs::read_to_string(repo_root().join(BASELINE_REL)).expect("read primary baseline"),
    )
    .expect("parse primary baseline");
    let spec = fs::read_to_string(repo_root().join(CONTROLLING_SPEC_REL)).unwrap();

    for registered in FINAL_NONNUMERIC_ROWS {
        let row = baseline
            .rows
            .iter()
            .find(|row| {
                row.kind == registered.kind
                    && row.id == registered.id
                    && row
                        .flags
                        .iter()
                        .map(String::as_str)
                        .eq(registered.flags.iter().copied())
            })
            .unwrap_or_else(|| panic!("missing migrated nonnumeric row: {registered:?}"));
        assert!(
            row.citation.is_empty(),
            "final row retained legacy debt: {row:?}"
        );
        assert_eq!(
            capacity_census_authority::classify_final_authority(
                &authority_surface(row),
                final_authority_registries(),
                &spec,
            ),
            Ok(capacity_census_authority::FinalAuthority::Nonnumeric)
        );
    }

    for registered in FINAL_TAGGED_TRANSPORT_ROWS {
        let row = baseline
            .rows
            .iter()
            .find(|row| {
                row.kind == registered.kind
                    && row.id == registered.id
                    && row
                        .flags
                        .iter()
                        .map(String::as_str)
                        .eq(registered.flags.iter().copied())
            })
            .unwrap_or_else(|| panic!("missing registered tagged-transport row: {registered:?}"));
        assert!(
            row.citation.is_empty(),
            "final row retained legacy debt: {row:?}"
        );
        assert_eq!(
            capacity_census_authority::classify_final_authority(
                &authority_surface(row),
                final_authority_registries(),
                &spec,
            ),
            Ok(capacity_census_authority::FinalAuthority::TaggedTransport)
        );
    }

    for registration in FINAL_NUMERIC_OPERATION_ROWS {
        let registered = registration.surface;
        let row = baseline
            .rows
            .iter()
            .find(|row| {
                row.kind == registered.kind
                    && row.id == registered.id
                    && row
                        .flags
                        .iter()
                        .map(String::as_str)
                        .eq(registered.flags.iter().copied())
            })
            .unwrap_or_else(|| panic!("missing registered numeric-operation row: {registered:?}"));
        assert!(
            row.citation.is_empty(),
            "final row retained legacy debt: {row:?}"
        );
        assert_eq!(
            capacity_census_authority::classify_final_authority(
                &authority_surface(row),
                final_authority_registries(),
                &spec,
            ),
            Ok(
                capacity_census_authority::FinalAuthority::NumericOperation {
                    atom: registration.atom,
                }
            )
        );
    }
    let tensor_shape_registration = FINAL_NUMERIC_OPERATION_ROWS
        .iter()
        .find(|registration| registration.surface.id.contains("chelis_tensor_shape"))
        .expect("tensor_shape remains registered");
    let tensor_shape = baseline
        .rows
        .iter()
        .find(|row| row.id == tensor_shape_registration.surface.id)
        .expect("tensor_shape remains discovered");
    assert!(tensor_shape.citation.is_empty());
    // chelis#1308 re-registered the tagged-carrier `chelis_tensor_shape`
    // successor under [05-OP-33]; the pre-rebase pin of [05-OP-7] named
    // the pre-tagged-ABI registration this row superseded.
    assert_eq!(
        capacity_census_authority::classify_final_authority(
            &authority_surface(tensor_shape),
            final_authority_registries(),
            &spec,
        ),
        Ok(capacity_census_authority::FinalAuthority::NumericOperation { atom: "[05-OP-33]" })
    );
}

#[test]
fn primary_baseline_has_zero_legacy_rows() {
    let baseline: Baseline = serde_json::from_str(
        &fs::read_to_string(repo_root().join(BASELINE_REL)).expect("read primary baseline"),
    )
    .expect("parse primary baseline");
    let legacy: Vec<&Row> = baseline
        .rows
        .iter()
        .filter(|row| !row.citation.trim().is_empty())
        .collect();
    assert!(legacy.is_empty(), "primary legacy rows remain: {legacy:#?}");
}

fn row_from_final(descriptor: StaticSurfaceDescriptor) -> Row {
    Row {
        kind: descriptor.kind.to_string(),
        id: descriptor.id.to_string(),
        flags: descriptor
            .flags
            .iter()
            .map(|flag| (*flag).to_string())
            .collect(),
        citation: String::new(),
    }
}

#[test]
fn duplicate_primary_descriptors_fail_before_map_collapse() {
    let row = row_from_final(FINAL_NONNUMERIC_ROWS[0]);
    let duplicate_baseline = Baseline {
        version: 3,
        legs: coverage_manifest(),
        rows: vec![row.clone(), row.clone()],
    };
    let error = check_against_baseline(std::slice::from_ref(&row), &duplicate_baseline)
        .expect_err("duplicate baseline identities must fail");
    assert!(error.contains("DUPLICATE BASELINE DESCRIPTOR"), "{error}");

    let baseline = Baseline {
        version: 3,
        legs: coverage_manifest(),
        rows: vec![row.clone()],
    };
    let error = check_against_baseline(&[row.clone(), row], &baseline)
        .expect_err("duplicate current identities must fail");
    assert!(error.contains("DUPLICATE CURRENT DESCRIPTOR"), "{error}");
}

#[test]
fn primary_metadata_and_manifest_mutations_fail() {
    let row = row_from_final(FINAL_NONNUMERIC_ROWS[0]);
    let baseline = Baseline {
        version: 3,
        legs: coverage_manifest(),
        rows: vec![row.clone()],
    };
    let mut changed = row.clone();
    changed.flags.push("numeric-op".to_string());
    let error = check_against_baseline(&[changed], &baseline)
        .expect_err("derived metadata changes must fail");
    assert!(error.contains("ENFORCEMENT METADATA CHANGED"), "{error}");

    let mut wrong_manifest = baseline;
    wrong_manifest.legs.covered[0]
        .command
        .push_str(" --weakened");
    let error = check_against_baseline(std::slice::from_ref(&row), &wrong_manifest)
        .expect_err("editable coverage claims must fail");
    assert!(error.contains("INVALID COVERAGE MANIFEST"), "{error}");
}

#[test]
fn every_primary_final_registration_binds_kind_identity_and_flags() {
    let spec = fs::read_to_string(repo_root().join(CONTROLLING_SPEC_REL)).unwrap();
    let registered: Vec<StaticSurfaceDescriptor> = FINAL_NONNUMERIC_ROWS
        .iter()
        .chain(FINAL_TAGGED_TRANSPORT_ROWS)
        .copied()
        .chain(
            FINAL_NUMERIC_OPERATION_ROWS
                .iter()
                .map(|registration| registration.surface),
        )
        .collect();

    for descriptor in registered {
        let exact = SurfaceDescriptor::new(
            descriptor.family,
            descriptor.kind,
            descriptor.id,
            descriptor.flags,
        );
        capacity_census_authority::classify_final_authority(
            &exact,
            final_authority_registries(),
            &spec,
        )
        .unwrap_or_else(|problem| panic!("registered descriptor must classify: {problem}"));

        let mut wrong_kind = exact.clone();
        wrong_kind.kind.push_str("-mutated");
        assert!(
            capacity_census_authority::classify_final_authority(
                &wrong_kind,
                final_authority_registries(),
                &spec,
            )
            .is_err(),
            "kind mutation inherited authority: {descriptor:?}"
        );

        let mut wrong_identity = exact.clone();
        wrong_identity.id.push_str(" mutated");
        assert!(
            capacity_census_authority::classify_final_authority(
                &wrong_identity,
                final_authority_registries(),
                &spec,
            )
            .is_err(),
            "identity mutation inherited authority: {descriptor:?}"
        );

        let mut wrong_flags = exact;
        wrong_flags.flags.push("float-carrier".to_string());
        assert!(
            capacity_census_authority::classify_final_authority(
                &wrong_flags,
                final_authority_registries(),
                &spec,
            )
            .is_err(),
            "flag mutation inherited authority: {descriptor:?}"
        );
    }
}

#[test]
fn regeneration_preserves_final_authority_but_cannot_bless_an_unclassified_row() {
    let final_row = Row {
        kind: FINAL_NONNUMERIC_ROWS[0].kind.to_string(),
        id: FINAL_NONNUMERIC_ROWS[0].id.to_string(),
        flags: Vec::new(),
        citation: String::new(),
    };
    let unclassified = Row {
        kind: "header-export".to_string(),
        id: "reviewer.h: void unclassified ( void ) ;".to_string(),
        flags: Vec::new(),
        citation: String::new(),
    };
    let path = std::env::temp_dir().join(format!(
        "chelis-capacity-census-regeneration-{}.json",
        std::process::id()
    ));
    regenerate(&path, &[final_row.clone(), unclassified.clone()], None);
    let baseline: Baseline = serde_json::from_str(
        &fs::read_to_string(&path).expect("read regenerated synthetic baseline"),
    )
    .expect("parse regenerated synthetic baseline");
    fs::remove_file(&path).ok();

    assert!(baseline.rows[0].citation.is_empty());
    assert_eq!(baseline.rows[1].citation, "TODO");
    let error = check_against_baseline(&[final_row, unclassified], &baseline)
        .expect_err("regeneration must leave the new row visibly unclassified");
    assert!(error.contains("UNCLASSIFIED census row"), "{error}");
}

#[test]
fn sanctioned_actions_name_complete_identity_change_and_new_atom_paths() {
    let guidance = teaching_footer();
    for required in [
        "removal plus addition",
        "STRUCTURALLY NONNUMERIC",
        "TAGGED TRANSPORT",
        "`NumericOperationRegistration`",
        "No transition exception",
        "highest allocated atom number",
        "scripts/generate_rejection_registries.py --write",
        "crates/chelis-types/src/rejection_registry_generated.rs",
    ] {
        assert!(
            guidance.contains(required),
            "the sanctioned-action guidance must name `{required}`:\n{guidance}"
        );
    }
}

// ---------------------------------------------------------------------------
// Planted-evasion unit tests (the negative parity for the guard itself)
// ---------------------------------------------------------------------------

#[test]
fn planted_dtype_int_export_is_flagged() {
    // The chelis#891 pad_sequences shape: a (value, int dtype) pair.
    let rows = header_rows_local(
        "planted.h",
        "chelis_tensor *chelis_pad_sequences(const chelis_list *sequences,\n\
         chelis_value pad_value, int pad_dtype);\n",
    );
    assert_eq!(rows.len(), 1, "planted export must be enumerated: {rows:?}");
    assert!(
        rows[0].flags.contains(&"raw-dtype-int".to_string()),
        "int pad_dtype must classify as raw-dtype-int: {rows:?}"
    );
}

#[test]
fn planted_multiline_and_float_carrier() {
    let rows = header_rows_local(
        "planted.h",
        "double chelis_read_scalar(\n    const chelis_tensor *t,\n    int index);\n",
    );
    assert_eq!(rows.len(), 1);
    assert!(rows[0].flags.contains(&"float-carrier".to_string()));
    assert!(
        !rows[0].flags.contains(&"raw-dtype-int".to_string()),
        "an int param without a dtype-ish name is inventoried but unflagged: {rows:?}"
    );
}

#[test]
fn planted_static_inline_carries_no_abi_row() {
    let rows = header_rows_local(
        "planted.h",
        "static inline float bits_to_f32(uint32_t b) { return 0.0f; }\n\
         void chelis_real_export(int x);\n",
    );
    let ids: Vec<&str> = rows.iter().map(|r| r.id.as_str()).collect();
    assert_eq!(
        ids,
        ["planted.h: void chelis_real_export ( int x ) ;"],
        "static inline must be skipped, the real export kept"
    );
}

#[test]
fn planted_struct_layout_is_inventoried() {
    let rows = header_rows_local(
        "planted.h",
        "typedef struct {\n  int dtype;\n  double f64_;\n} planted_value;\n",
    );
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert_eq!(rows[0].kind, "header-struct");
    assert!(rows[0].flags.contains(&"float-carrier".to_string()));
    assert!(rows[0].flags.contains(&"raw-dtype-int".to_string()));
}

#[test]
fn planted_deftype_with_f64_variant_is_detected() {
    // A complete, canonical declaration reaches the resolver before the
    // census follows the field's numeric payload.
    let mut rows = Vec::new();
    scan_deftypes(&[planted_numeric_adt("JNum")], "planted", &mut rows);
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert_eq!(
        rows[0].id,
        "planted::Json: () (variant {} JNum (t-prim {} f64))"
    );
}

#[test]
fn todo_citation_fails_with_teaching_message() {
    let row = Row {
        kind: "header-export".to_string(),
        id: "planted.h: void x(void);".to_string(),
        flags: vec![],
        citation: "TODO".to_string(),
    };
    let baseline = Baseline {
        version: 3,
        legs: coverage_manifest(),
        rows: vec![row.clone()],
    };
    let current = vec![row];
    let err = check_against_baseline(&current, &baseline).unwrap_err();
    assert!(err.contains("UNCLASSIFIED census row"), "{err}");
    assert!(
        err.contains("spec/design/dtype_semantics.md §C6")
            && err.contains("AGENTS.md §Numeric Surface Discipline")
            && err.contains("STRUCTURALLY NONNUMERIC")
            && err.contains("TAGGED TRANSPORT")
            && err.contains("NUMERIC OPERATION"),
        "the failure message must teach the rule and the sanctioned actions: {err}"
    );
}

#[test]
fn new_and_removed_rows_fail() {
    let cited = |id: &str| Row {
        kind: "header-export".to_string(),
        id: id.to_string(),
        flags: vec![],
        citation: "baseline-2026-07-30".to_string(),
    };
    let baseline = Baseline {
        version: 3,
        legs: coverage_manifest(),
        rows: vec![cited("a.h: void old(void);")],
    };
    let current = vec![cited("a.h: void brand_new(void);")];
    let err = check_against_baseline(&current, &baseline).unwrap_err();
    assert!(err.contains("NEW surface"), "{err}");
    assert!(err.contains("REMOVED surface"), "{err}");
    assert!(err.contains("anti-churn invariant 7"), "{err}");
}

fn flagged_row(id: &str, citation: &str) -> Row {
    Row {
        kind: "header-export".to_string(),
        id: id.to_string(),
        flags: vec!["raw-dtype-int".to_string()],
        citation: citation.to_string(),
    }
}

/// PR #950 red team P1-1: opening a fresh issue and citing it must NOT
/// bless a new capacity seam - flagged rows have no issue-citation path.
#[test]
fn new_flagged_seam_cannot_be_cited_with_an_issue() {
    let row = flagged_row(
        "planted.h: void f(chelis_value v, int pad_dtype);",
        "chelis#123456",
    );
    let baseline = Baseline {
        version: 3,
        legs: coverage_manifest(),
        rows: vec![row.clone()],
    };
    let err = check_against_baseline(&[row], &baseline).unwrap_err();
    assert!(
        err.contains("UNCLASSIFIED census row"),
        "an issue citation must not bless a flagged row: {err}"
    );
}

/// A human override is not one of the three final authority classes.
#[test]
fn maintainer_override_is_not_a_final_authority_class() {
    let row = flagged_row(
        "planted.h: void staged(int out_dtype);",
        "maintainer-override(FFI staging for chelis#893, chelis#893)",
    );
    let baseline = Baseline {
        version: 3,
        legs: coverage_manifest(),
        rows: vec![row.clone()],
    };
    let error = check_against_baseline(&[row], &baseline)
        .expect_err("generic maintainer overrides are transition debt, not final authority");
    assert!(error.contains("UNCLASSIFIED census row"), "{error}");
}

#[test]
fn maintainer_override_does_not_waive_new_numeric_semantic_registration() {
    let mut row =
        header_rows_local("planted.h", "double staged(double value, int out_dtype);").remove(0);
    row.citation = "maintainer-override(FFI staging, chelis#893)".to_string();
    let baseline = Baseline {
        version: 3,
        legs: coverage_manifest(),
        rows: vec![row.clone()],
    };
    let err = check_against_baseline(&[row], &baseline).unwrap_err();
    assert!(
        err.contains("UNCLASSIFIED census row"),
        "a generic override is not one of the three final classes: {err}"
    );
}

/// Arbitrary prose cannot classify a row.
#[test]
fn prose_citation_without_issue_ref_fails() {
    let row = Row {
        kind: "header-export".to_string(),
        id: "planted.h: void plain(chelis_string s);".to_string(),
        flags: vec![],
        citation: "reviewed and fine".to_string(),
    };
    let baseline = Baseline {
        version: 3,
        legs: coverage_manifest(),
        rows: vec![row.clone()],
    };
    let err = check_against_baseline(&[row], &baseline).unwrap_err();
    assert!(err.contains("UNCLASSIFIED census row"), "{err}");
}

/// An override spelling cannot classify a row, with or without an issue.
#[test]
fn maintainer_override_without_issue_ref_fails() {
    let row = flagged_row(
        "planted.h: void staged(int out_dtype);",
        "maintainer-override(because I said so)",
    );
    let baseline = Baseline {
        version: 3,
        legs: coverage_manifest(),
        rows: vec![row.clone()],
    };
    let err = check_against_baseline(&[row], &baseline).unwrap_err();
    assert!(err.contains("UNCLASSIFIED census row"), "{err}");
}

/// Empty numeric flags are not proof of structural nonnumericity, and an
/// issue citation is not a fourth final authority class.
#[test]
fn unflagged_row_with_issue_citation_still_requires_final_authority() {
    let row = Row {
        kind: "header-export".to_string(),
        id: "planted.h: void plain(chelis_string s);".to_string(),
        flags: vec![],
        citation: "chelis#123456".to_string(),
    };
    let baseline = Baseline {
        version: 3,
        legs: coverage_manifest(),
        rows: vec![row.clone()],
    };
    let error = check_against_baseline(&[row], &baseline)
        .expect_err("an issue citation cannot classify an unflagged descriptor");
    assert!(error.contains("UNCLASSIFIED census row"), "{error}");
}

// ---------------------------------------------------------------------------
// The re-red-team's four executed mutations (PR #950, 2026-07-30), kept as
// standing negative tests.
// ---------------------------------------------------------------------------

/// A macro-hidden float carrier must be visible: the census consumes the
/// REAL preprocessor's output, so the spelling arrives expanded.
#[test]
fn reviewer_preprocessor_capacity_seam_is_visible() {
    let dir = std::env::temp_dir().join(format!("census-pp-{}", std::process::id()));
    fs::create_dir_all(&dir).expect("temp include dir");
    fs::write(
        dir.join("planted.h"),
        "#define CHELIS_NUM double\n\
         CHELIS_NUM chelis_macro_result(int64_t x);\n",
    )
    .expect("write planted header");
    let per_file = preprocessed_headers(&dir, &["planted.h"]);
    let text = per_file.get("planted.h").expect("planted attributed");
    let rows = header_rows("planted.h", text, &collect_typedefs(text));
    let row = rows
        .iter()
        .find(|r| r.id.contains("chelis_macro_result"))
        .expect("export enumerated");
    assert!(
        row.flags.iter().any(|f| f == "float-carrier"),
        "the preprocessed declaration carries double: {row:?}"
    );
    fs::remove_dir_all(&dir).ok();
}

/// A typedef-hidden raw dtype int must be visible: classification resolves
/// typedef spellings before matching.
#[test]
fn reviewer_typedef_capacity_seam_is_visible() {
    let rows = header_rows_local(
        "planted.h",
        "typedef int chelis_dtype_id;\n\
         void chelis_typedef_dtype(chelis_value value, chelis_dtype_id dtype);\n",
    );
    let row = rows
        .iter()
        .find(|r| r.id.contains("chelis_typedef_dtype"))
        .expect("export enumerated");
    assert!(
        row.flags.iter().any(|f| f == "raw-dtype-int"),
        "the typedef resolves to raw int: {row:?}"
    );
}

/// A new post-ratchet numeric runtime export (exact-width types, so not a
/// capacity seam) cannot enter with only a tracker citation: it needs its
/// spec/05 semantic registration in the same change set.
#[test]
fn new_post_ratchet_runtime_numeric_op_requires_semantic_registration() {
    let mut rows = header_rows_local("planted.h", "int64_t chelis_abs_i64(int64_t value);");
    assert_eq!(rows.len(), 1);
    assert!(!is_seam(&rows[0].flags), "exact-width types are not seams");
    assert!(
        rows[0].flags.iter().any(|f| f == "numeric-op"),
        "numeric-op membership is structural: {:?}",
        rows[0]
    );
    let row = rows.remove(0);
    let regenerated = Baseline {
        version: 3,
        legs: coverage_manifest(),
        rows: vec![row.clone()],
    };
    let err = check_against_baseline(std::slice::from_ref(&row), &regenerated).unwrap_err();
    assert!(
        err.contains("UNCLASSIFIED census row"),
        "a new post-ratchet numeric callable needs a semantic decision, not a tracker citation: {err}"
    );
    let registered = row;
    let ok_baseline = Baseline {
        version: 3,
        legs: coverage_manifest(),
        rows: vec![registered.clone()],
    };
    let registration = FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "header-export",
            "planted.h: int64_t chelis_abs_i64 ( int64_t value ) ;",
            &["numeric-op"],
        ),
        atom: "[05-OP-1]",
        authority_anchor: "Integer abs",
    };
    assert!(
        check_against_baseline_with_authorities(
            &[registered],
            &ok_baseline,
            &[],
            "Synthetic controlling fixture:\n> **[05-OP-1]** Integer abs.",
            AuthorityRegistries {
                nonnumeric: &[],
                tagged_transports: &[],
                numeric_operations: &[registration],
            },
        )
        .is_ok()
    );
}

// ---------------------------------------------------------------------------
// PR #956 correction tests (2026-07-30): these lock the exact omissions found
// by the exact-head review and the failed Linux Integration run.
// ---------------------------------------------------------------------------

#[test]
fn c_identity_is_token_canonical_across_preprocessor_whitespace() {
    let apple = header_rows_local(
        "planted.h",
        "chelis_string chelis_string_from_bool(_Bool value);\n",
    );
    let linux = header_rows_local(
        "planted.h",
        "chelis_string chelis_string_from_bool( _Bool value);\n",
    );
    assert_eq!(
        apple, linux,
        "C token identity must not depend on a preprocessor's whitespace rendering"
    );
}

#[test]
fn c_identity_is_stable_across_bool_preprocessor_spellings() {
    let c23 = header_rows_local(
        "planted.h",
        "chelis_value chelis_value_from_bool(bool value);\n",
    );
    let legacy = header_rows_local(
        "planted.h",
        "chelis_value chelis_value_from_bool(_Bool value);\n",
    );
    assert_eq!(
        c23, legacy,
        "C23 `bool` and legacy `_Bool` must have one census identity"
    );
}

#[test]
fn unrelated_observation_atom_is_not_a_numeric_registration() {
    let mut rows = header_rows_local("planted.h", "int64_t chelis_abs_i64(int64_t value);");
    let mut row = rows.remove(0);
    row.citation = "chelis#729; spec/05-risc-primitives.md [05-OBS-1]".to_string();
    let baseline = Baseline {
        version: 3,
        legs: coverage_manifest(),
        rows: vec![row.clone()],
    };
    let registration = SemanticRegistration {
        callable: "[header-export] planted.h: int64_t chelis_abs_i64 ( int64_t value ) ;",
        atom: "[05-OBS-1]",
    };
    let spec = fs::read_to_string(repo_root().join(CONTROLLING_SPEC_REL)).unwrap();
    let err = check_against_baseline_with(&[row], &baseline, &[registration], &spec).unwrap_err();
    assert!(
        err.contains("INVALID SEMANTIC REGISTRATION")
            && err.contains("wrong atom grammar/group")
            && err.contains("05-OBS-1"),
        "an unrelated existing atom must not bless a callable: {err}"
    );
}

#[test]
fn nonexistent_operation_atom_is_not_a_numeric_registration() {
    let row = header_rows_local("planted.h", "int64_t chelis_abs_i64(int64_t value);").remove(0);
    let baseline = Baseline {
        version: 3,
        legs: coverage_manifest(),
        rows: vec![row.clone()],
    };
    let registration = SemanticRegistration {
        callable: "[header-export] planted.h: int64_t chelis_abs_i64 ( int64_t value ) ;",
        atom: "[05-OP-999]",
    };
    let err =
        check_against_baseline_with(&[row], &baseline, &[registration], "no op atoms").unwrap_err();
    assert!(
        err.contains("does not exist as a normative") && err.contains("[05-OP-999]"),
        "a syntactically valid but absent atom must fail: {err}"
    );
}

#[test]
fn operation_atom_cross_reference_is_not_a_normative_definition() {
    let mut row =
        header_rows_local("planted.h", "int64_t chelis_abs_i64(int64_t value);").remove(0);
    row.citation = "chelis#729".to_string();
    let baseline = Baseline {
        version: 3,
        legs: coverage_manifest(),
        rows: vec![row.clone()],
    };
    let registration = SemanticRegistration {
        callable: "[header-export] planted.h: int64_t chelis_abs_i64 ( int64_t value ) ;",
        atom: "[05-OP-1]",
    };
    let err = check_against_baseline_with(
        &[row],
        &baseline,
        &[registration],
        "A cross-reference to [05-OP-1] is not its definition.",
    )
    .unwrap_err();
    assert!(
        err.contains("does not exist as a normative"),
        "a textual mention must not satisfy atom existence: {err}"
    );
}

#[test]
fn old_style_semantic_registration_and_issue_prose_are_not_final_authority() {
    let mut row =
        header_rows_local("planted.h", "int64_t chelis_abs_i64(int64_t value);").remove(0);
    row.citation = "reviewed semantic registration".to_string();
    let baseline = Baseline {
        version: 3,
        legs: coverage_manifest(),
        rows: vec![row.clone()],
    };
    let registration = SemanticRegistration {
        callable: "[header-export] planted.h: int64_t chelis_abs_i64 ( int64_t value ) ;",
        atom: "[05-OP-1]",
    };
    let err = check_against_baseline_with(
        &[row],
        &baseline,
        &[registration],
        "> **[05-OP-1]** Synthetic exact callable semantics.",
    )
    .unwrap_err();
    assert!(
        err.contains("UNCLASSIFIED census row"),
        "only a family-qualified complete final registration authorizes a discovered row: {err}"
    );
}

fn planted_numeric_adt(variant_name: &str) -> Expr {
    let span = chelis_deep::Span::new(0, 0);
    let tprim = Expr::node(
        DeepTag::TPrim,
        Default::default(),
        vec![Expr::Atom(Atom::Name("f64".to_string()), span)],
        span,
    );
    let variant = Expr::node(
        DeepTag::Variant,
        Default::default(),
        vec![
            Expr::Atom(Atom::Name(variant_name.to_string()), span),
            tprim,
        ],
        span,
    );
    Expr::node(
        DeepTag::Deftype,
        Default::default(),
        vec![
            Expr::Atom(Atom::Name("Json".to_string()), span),
            Expr::List(List { elements: vec![] }, span),
            variant,
        ],
        span,
    )
}

#[test]
fn std_adt_identity_changes_when_same_dtype_variant_changes() {
    let mut before = Vec::new();
    let mut after = Vec::new();
    scan_deftypes(&[planted_numeric_adt("JNum")], "planted", &mut before);
    scan_deftypes(&[planted_numeric_adt("JNumber")], "planted", &mut after);
    assert_eq!(before.len(), 1);
    assert_eq!(after.len(), 1);
    assert_ne!(
        before[0].id, after[0].id,
        "variant and field shape, not only the dtype set, controls ADT identity"
    );
}

#[test]
fn exported_public_numeric_stdlib_def_is_enumerated() {
    let decls = chelis_surf::parser::parse_str(
        "module Planted\n\
         export (public_numeric)\n\
         def public_numeric(x: int64) -> int64 = x\n",
    )
    .expect("planted stdlib source parses");
    let exprs = chelis_surf::desugar::desugar_program(&decls);
    let mut rows = Vec::new();
    scan_deftypes(&exprs, "planted", &mut rows);
    assert!(
        rows.iter()
            .any(|row| { row.kind == "std-def-numeric" && row.id.contains("public_numeric") }),
        "every exported numeric stdlib def must have a census row: {rows:?}"
    );
}

/// Write a throwaway published-include directory. Stale content is cleared
/// first because the derived-roots assertion compares against the directory
/// walk. A name may carry a subdirectory (`sub/x.h`), which is how the
/// recursive-walk controls plant a header one level down.
fn planted_include_dir(label: &str, files: &[(&str, &str)]) -> PathBuf {
    let sequence = PLANTED_INCLUDE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let dir =
        std::env::temp_dir().join(format!("census-{label}-{}-{sequence}", std::process::id()));
    fs::create_dir_all(&dir).expect("temp include dir");
    for (name, body) in files {
        let path = dir.join(name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("planted header parent");
        }
        fs::write(&path, body).expect("write planted header");
    }
    dir
}

/// Run a census guard expected to reject its input and return the teaching
/// message. The message IS the contract for a context-poor agent, so every
/// rejection test asserts on its text.
fn expect_census_panic(guard: impl FnOnce() + std::panic::UnwindSafe) -> String {
    let panic = std::panic::catch_unwind(guard).expect_err("the census guard must reject this");
    if let Some(message) = panic.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = panic.downcast_ref::<&str>() {
        (*message).to_string()
    } else {
        String::new()
    }
}

#[test]
fn conditional_public_abi_is_mechanically_rejected() {
    let dir = planted_include_dir(
        "context",
        &[(
            "planted.h",
            "#ifdef PLANTED_WIDE\n\
             double chelis_contextual(double x);\n\
             #else\n\
             int64_t chelis_contextual(int64_t x);\n\
             #endif\n",
        )],
    );
    let message = expect_census_panic(|| {
        preprocessed_headers(&dir, &["planted.h"]);
    });
    fs::remove_dir_all(&dir).ok();
    assert!(
        message.contains("CONTEXT-VARYING PUBLIC ABI"),
        "the rejection must teach the totality rule: {message}"
    );
}

#[test]
fn shared_header_cannot_have_multiple_public_macro_contexts() {
    let dir = planted_include_dir(
        "root-context",
        &[
            (
                "shared.h",
                "CHELIS_NUM chelis_context_result(CHELIS_NUM value);\n",
            ),
            ("a.h", "#define CHELIS_NUM double\n#include \"shared.h\"\n"),
            ("b.h", "#define CHELIS_NUM float\n#include \"shared.h\"\n"),
        ],
    );
    let message = expect_census_panic(|| {
        preprocessed_headers(&dir, &["a.h", "b.h"]);
    });
    fs::remove_dir_all(&dir).ok();
    assert!(
        message.contains("CONTEXT-VARYING PUBLIC ABI"),
        "the rejection must name the context-invariance policy: {message}"
    );
}

#[test]
fn wire_coverage_manifest_names_graph_and_execution_authority() {
    let manifest = coverage_manifest();
    let wire = manifest
        .covered
        .iter()
        .find(|leg| leg.leg == "wire-schema-numeric-fields")
        .expect("the wire leg has an executable aggregate");
    assert!(wire.artifact.contains("publication roots"));
    assert!(wire.enumerator.contains("final authority"));
    assert!(wire.enumerator.contains("execution receipts"));
    assert_eq!(
        wire.command,
        "cargo nextest run -p chelis-compiler-api --test capacity_census_wire"
    );
    assert!(wire.mutations.iter().any(|test| {
        test == "verified_wire_authority_cannot_be_replaced_by_a_descriptor_or_baseline"
    }));
    assert!(
        wire.mutations
            .iter()
            .any(|test| { test == "wire_baseline_rejects_exception_fields_and_nonfinal_classes" })
    );
    assert!(
        wire.mutations
            .iter()
            .any(|test| { test == "wire_rows_cannot_erase_capacity_or_duplicate_an_identity" })
    );
    let baseline = Baseline {
        version: 3,
        legs: manifest,
        rows: vec![],
    };
    check_against_baseline(&[], &baseline).expect("the executable manifest is accepted");
}

#[test]
fn wire_coverage_cannot_drop_execution_or_restore_schema_only_inventory() {
    for change in ["artifact", "enumerator", "mutations"] {
        let mut manifest = coverage_manifest();
        let wire = manifest
            .covered
            .iter_mut()
            .find(|leg| leg.leg == "wire-schema-numeric-fields")
            .unwrap();
        match change {
            "artifact" => {
                wire.artifact =
                    "crates/chelis-compiler-api/src/schema.rs public serialized type graph"
                        .to_string();
            }
            "enumerator" => {
                wire.enumerator =
                    "rustdoc JSON public schema type graph -> wire-schema numeric fields"
                        .to_string();
            }
            "mutations" => {
                wire.mutations = vec![
                    "adding_or_removing_a_public_serialized_f64_field_changes_the_census"
                        .to_string(),
                ];
            }
            _ => unreachable!(),
        }
        let baseline = Baseline {
            version: 3,
            legs: manifest,
            rows: vec![],
        };
        let error = check_against_baseline(&[], &baseline)
            .expect_err("an obsolete wire manifest must fail");
        assert!(
            error.contains("INVALID COVERAGE MANIFEST"),
            "{change}: {error}"
        );
    }
}

#[test]
fn conditional_macro_taint_across_include_closure_is_rejected() {
    let dir = planted_include_dir(
        "closure-taint",
        &[
            (
                "root.h",
                "#ifdef CHELIS_REVIEW_WIDE\n\
                 #define CHELIS_NUM double\n\
                 #else\n\
                 #define CHELIS_NUM float\n\
                 #endif\n\
                 #include \"shared.h\"\n",
            ),
            (
                "shared.h",
                "CHELIS_NUM chelis_context_result(CHELIS_NUM value);\n",
            ),
        ],
    );
    let message = expect_census_panic(|| {
        preprocessed_headers(&dir, &["root.h"]);
    });
    fs::remove_dir_all(&dir).ok();
    assert!(
        message.contains("CONTEXT-VARYING PUBLIC ABI") && message.contains("CHELIS_NUM"),
        "the closure-wide rejection must name the tainted macro: {message}"
    );
}

#[test]
fn new_post_ratchet_bare_int_export_is_numeric_op_and_requires_registration() {
    let mut row = header_rows_local("planted.h", "int chelis_abs(int value);").remove(0);
    assert!(
        row.flags.iter().any(|flag| flag == "numeric-op"),
        "a by-value C int callable is conservatively numeric: {row:?}"
    );
    row.citation = "chelis#729".to_string();
    let baseline = Baseline {
        version: 3,
        legs: coverage_manifest(),
        rows: vec![row.clone()],
    };
    let err = check_against_baseline(&[row], &baseline).unwrap_err();
    assert!(
        err.contains("UNCLASSIFIED census row"),
        "a new post-ratchet bare-int numeric callable owes exact registration: {err}"
    );
}

// ---------------------------------------------------------------------------
// Round-3 red team (2026-07-31): the remaining silent-invisibility channels in
// the header leg, the unflagged stdlib-ADT family, and the two unvalidated
// citation strings. Each test below is that pass's executed probe, kept as a
// standing negative.
// ---------------------------------------------------------------------------

/// A declaration bracketed by `#line` directives is real, callable ABI that
/// the census attributes to a file outside the include directory and drops.
#[test]
fn line_directive_in_a_published_header_is_rejected() {
    let dir = planted_include_dir(
        "line-spoof",
        &[(
            "planted.h",
            "void chelis_visible(int x);\n\
             #line 1 \"/opt/vendor/x.h\"\n\
             double chelis_hidden(double value, int out_dtype);\n\
             #line 4 \"planted.h\"\n",
        )],
    );
    let message = expect_census_panic(|| {
        preprocessed_headers(&dir, &["planted.h"]);
    });
    fs::remove_dir_all(&dir).ok();
    assert!(
        message.contains("LINE-DIRECTIVE SPOOFING SURFACE") && message.contains("planted.h"),
        "a #line directive redirects the attribution the census reads back, \
         so it is banned outright: {message}"
    );
}

/// The backstop that does not depend on having enumerated the spoofing
/// channels: whatever the raw closure declares must reappear in some
/// attributed bucket.
#[test]
fn a_declaration_missing_from_every_attributed_bucket_fails() {
    let sources = BTreeMap::from([(
        "planted.h".to_string(),
        "void chelis_attributed(int x);\n\
         double chelis_orphaned(double value, int out_dtype);\n"
            .to_string(),
    )]);
    let complete = BTreeMap::from([(
        "planted.h".to_string(),
        "void chelis_attributed(int x);\n\
         double chelis_orphaned(double value, int out_dtype);\n"
            .to_string(),
    )]);
    assert_total_attribution(&sources, &complete);

    let truncated = BTreeMap::from([(
        "planted.h".to_string(),
        "void chelis_attributed(int x);\n".to_string(),
    )]);
    let message = expect_census_panic(|| {
        assert_total_attribution(&sources, &truncated);
    });
    assert!(
        message.contains("UNATTRIBUTED PUBLISHED DECLARATION")
            && message.contains("chelis_orphaned"),
        "attribution must be total over the raw published closure: {message}"
    );
}

/// Macro expansion legally rewrites type spellings between the raw and
/// preprocessed forms, so attribution totality compares declarator NAMES.
#[test]
fn attribution_totality_survives_legal_macro_expansion() {
    let sources = BTreeMap::from([(
        "planted.h".to_string(),
        "#define CHELIS_NUM double\nCHELIS_NUM chelis_macro_result(int64_t x);\n".to_string(),
    )]);
    let expanded = BTreeMap::from([(
        "planted.h".to_string(),
        "double chelis_macro_result(int64_t x);\n".to_string(),
    )]);
    assert_total_attribution(&sources, &expanded);
}

/// `AGENTS.md` states that a public ADT variant carrying bare `f64` has NO
/// citation path. `scan_deftypes` hard-coded empty flags, so the `io/json`
/// instance §C6 itself cites could land by regenerating and citing an open
/// issue - the round-1 P1-1 shape, closed for the header family only.
#[test]
fn std_adt_bare_f64_variant_has_no_issue_citation_path() {
    let mut rows = Vec::new();
    scan_deftypes(&[planted_numeric_adt("JsonBigNum")], "io/json", &mut rows);
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert!(
        rows[0].flags.iter().any(|flag| flag == "float-carrier"),
        "a public ADT variant carrying bare f64 is a capacity seam: {rows:?}"
    );
    let mut row = rows.remove(0);
    row.citation = "chelis#891".to_string();
    let baseline = Baseline {
        version: 3,
        legs: coverage_manifest(),
        rows: vec![row.clone()],
    };
    let err = check_against_baseline(&[row], &baseline).unwrap_err();
    assert!(
        err.contains("UNCLASSIFIED census row"),
        "regenerate-and-cite must not land a new f64 ADT channel: {err}"
    );
}

/// The other half of the ADT rule: an integer carrier is a numeric channel,
/// not a seam with no citation path. A new post-ratchet descriptor with this
/// classification requires exact semantic registration.
#[test]
fn std_adt_integer_carrier_is_numeric_op_not_a_seam() {
    let span = chelis_deep::Span::new(0, 0);
    let tprim = Expr::node(
        DeepTag::TPrim,
        Default::default(),
        vec![Expr::Atom(Atom::Name("int64".to_string()), span)],
        span,
    );
    let variant = Expr::node(
        DeepTag::Variant,
        Default::default(),
        vec![Expr::Atom(Atom::Name("JsonInt".to_string()), span), tprim],
        span,
    );
    let deftype = Expr::node(
        DeepTag::Deftype,
        Default::default(),
        vec![
            Expr::Atom(Atom::Name("Json".to_string()), span),
            Expr::List(List { elements: vec![] }, span),
            variant,
        ],
        span,
    );
    let mut rows = Vec::new();
    scan_deftypes(&[deftype], "io/json", &mut rows);
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert!(
        !is_seam(&rows[0].flags) && rows[0].flags.iter().any(|flag| flag == "numeric-op"),
        "a source-faithful integer variant is the WANTED shape - classified, \
         not seamed: {rows:?}"
    );
}

/// `extern <type> <name>;` is published ABI with no callable to classify,
/// and produced no row at all before this.
#[test]
fn extern_data_declarations_are_inventoried_and_classified() {
    let rows = header_rows_local(
        "planted.h",
        "extern double chelis_global_scale;\nextern int chelis_default_dtype;\n",
    );
    assert_eq!(rows.len(), 2, "both data declarations are ABI: {rows:?}");
    assert!(
        rows.iter().all(|row| row.kind == "header-data"),
        "non-function ABI has its own kind: {rows:?}"
    );
    let scale = rows
        .iter()
        .find(|row| row.id.contains("chelis_global_scale"))
        .expect("float global enumerated");
    assert!(
        scale.flags.iter().any(|flag| flag == "float-carrier"),
        "an exported bare double global is a capacity seam: {scale:?}"
    );
    let dtype = rows
        .iter()
        .find(|row| row.id.contains("chelis_default_dtype"))
        .expect("dtype global enumerated");
    assert!(
        dtype.flags.iter().any(|flag| flag == "raw-dtype-int"),
        "an exported raw dtype id is a capacity seam: {dtype:?}"
    );

    let mut cited = scale.clone();
    cited.citation = "chelis#729".to_string();
    let baseline = Baseline {
        version: 3,
        legs: coverage_manifest(),
        rows: vec![cited.clone()],
    };
    let err = check_against_baseline(&[cited], &baseline).unwrap_err();
    assert!(
        err.contains("UNCLASSIFIED census row"),
        "data seams take the same no-citation-path rule as callables: {err}"
    );
}

/// A callback typedef hides its return and parameter spellings behind an
/// opaque name, so the setter taking it was inventoried with EMPTY flags.
#[test]
fn function_pointer_typedef_cannot_launder_a_seam() {
    let rows = header_rows_local(
        "planted.h",
        "typedef double (*chelis_elem_cb)(double value, int elem_dtype);\n\
         void chelis_set_elem_cb(chelis_elem_cb callback);\n",
    );
    let row = rows
        .iter()
        .find(|row| row.id.contains("chelis_set_elem_cb"))
        .expect("the setter is enumerated");
    assert!(
        row.flags.iter().any(|flag| flag == "float-carrier"),
        "the callback's double return/parameter reaches the setter: {row:?}"
    );
    assert!(
        row.flags.iter().any(|flag| flag == "raw-dtype-int"),
        "the callback's raw dtype parameter reaches the setter: {row:?}"
    );
}

/// Typedef resolution stays TOTAL: a parenthesized typedef the resolver
/// does not understand is rejected rather than skipped, because skipping it
/// is exactly how the function-pointer seam laundered itself.
#[test]
fn unresolvable_parenthesized_typedef_is_rejected() {
    let message = expect_census_panic(|| {
        header_rows_local("planted.h", "typedef double (chelis_weird)[4];\n");
    });
    assert!(
        message.contains("UNRESOLVABLE PARENTHESIZED TYPEDEF"),
        "an unresolvable typedef must fail loudly, not silently: {message}"
    );
}

/// Reachability comes from the raw include graph, even when the real
/// preprocessor emits no local declaration bucket for a root. This is the
/// Linux `chelis_blas.h` shape: its only emitted content can be system ABI.
#[test]
fn a_declaration_free_root_is_still_reached() {
    let dir = planted_include_dir(
        "declaration-free-root",
        &[("root.h", "#include <stddef.h>\n")],
    );
    let per_file = preprocessed_headers(&dir, &["root.h"]);
    fs::remove_dir_all(&dir).ok();
    assert!(
        per_file
            .iter()
            .flat_map(|(name, text)| header_rows_local(name, text))
            .next()
            .is_none(),
        "the root is reached even though it contributes no ABI row: {per_file:?}"
    );
}

/// Negative parity for §C6's derived-root rule: a header absent from every
/// root's raw include closure is not reached merely because it exists on disk.
#[test]
fn a_published_header_reachable_from_no_root_fails() {
    let dir = planted_include_dir(
        "orphan-header",
        &[
            ("root.h", "void chelis_rooted(int x);\n"),
            (
                "orphan.h",
                "double chelis_orphan(double value, int out_dtype);\n",
            ),
        ],
    );
    let message = expect_census_panic(|| {
        preprocessed_headers(&dir, &["root.h"]);
    });
    assert!(
        message.contains("PUBLISHED HEADER NOT REACHED FROM ANY ROOT")
            && message.contains("orphan.h"),
        "an unreachable published header must fail, not vanish: {message}"
    );

    let per_file = preprocessed_headers(&dir, &["root.h", "orphan.h"]);
    fs::remove_dir_all(&dir).ok();
    assert_eq!(
        per_file.keys().cloned().collect::<Vec<_>>(),
        ["orphan.h", "root.h"],
        "declaring both roots reaches the whole published directory"
    );
}

/// `cc -E -I <dir>` resolves `<x>` against the include path exactly as it
/// resolves `"x"`, so a raw-source scan that follows only quoted includes
/// leaves an angle-included local header outside every raw-source guard.
#[test]
fn angle_included_local_header_is_inside_the_context_guard() {
    let dir = planted_include_dir(
        "angle-include",
        &[
            ("root.h", "#include <sub.h>\n"),
            (
                "sub.h",
                "#ifdef PLANTED_WIDE\n\
                 double chelis_angle(double x);\n\
                 #else\n\
                 int64_t chelis_angle(int64_t x);\n\
                 #endif\n",
            ),
        ],
    );
    let message = expect_census_panic(|| {
        preprocessed_headers(&dir, &["root.h"]);
    });
    fs::remove_dir_all(&dir).ok();
    assert!(
        message.contains("CONTEXT-VARYING PUBLIC ABI") && message.contains("sub.h"),
        "an angle-included local header is inside the closure: {message}"
    );
}

#[test]
fn local_include_reads_both_spellings_and_nothing_else() {
    assert_eq!(
        local_include("#include \"chelis_simd.h\""),
        Some("chelis_simd.h")
    );
    assert_eq!(local_include("  #include <sub.h>"), Some("sub.h"));
    assert_eq!(local_include("#define CHELIS_NUM double"), None);
    assert_eq!(local_include("void f(void);"), None);
}

/// The generic override parser is gone. No spelling of the old marker is a
/// final authority class, including the formerly well-formed shape.
#[test]
fn no_generic_maintainer_override_spelling_is_authority() {
    for citation in [
        "maintainer-override(FFI staging chelis#893",
        "maintainer-override() chelis#893",
        "maintainer-override(FFI staging) chelis#893",
        "maintainer-override(FFI staging (temporary) for chelis#893)",
    ] {
        let row = flagged_row("planted.h: void staged(int out_dtype);", citation);
        let baseline = Baseline {
            version: 3,
            legs: coverage_manifest(),
            rows: vec![row.clone()],
        };
        let err = check_against_baseline(&[row], &baseline).unwrap_err();
        assert!(
            err.contains("UNCLASSIFIED census row"),
            "generic override must be rejected: {err}"
        );
    }
}

#[test]
fn invented_permanent_disposition_is_not_a_citation() {
    let row = Row {
        kind: "header-export".to_string(),
        id: "reviewer.h: void stable(void);".to_string(),
        flags: Vec::new(),
        citation: "permanent-disposition(reviewed and fine)".to_string(),
    };
    let baseline = Baseline {
        version: 3,
        legs: coverage_manifest(),
        rows: vec![row.clone()],
    };
    let err = check_against_baseline_with(&[row], &baseline, &[], "")
        .expect_err("an invented permanent disposition must fail");
    assert!(
        err.contains("UNCLASSIFIED census row"),
        "unexpected invented-disposition diagnostic: {err}"
    );
}

// ---------------------------------------------------------------------------
// Round-4 red team (2026-07-31) closures. Each of these locks a claim the
// round proved was either false or untested; the guard-revert mutation that
// makes each one RED is named in its doc comment.
// ---------------------------------------------------------------------------

/// N1, the finding that mattered: `NUMERIC_C_TYPES` was an ALLOWLIST, so an
/// arithmetic spelling it had never heard of classified as `[]` and a bare
/// 16-bit float export could enter the census on an ordinary issue
/// citation - past the rule that flagged rows have no citation path. The
/// rule is now inverted, and these are the exact spellings the round team
/// executed end-to-end.
///
/// Reverting `assert_known_type_words` to a no-op turns this RED.
#[test]
fn an_unknown_c_type_word_is_rejected_rather_than_classified_empty() {
    for spelling in ["_Float16", "__fp16", "__bf16", "_Decimal64", "__int128"] {
        let dir = planted_include_dir(
            "unknown-type-word",
            &[("root.h", &format!("{spelling} chelis_widen(int value);\n"))],
        );
        let message = expect_census_panic(|| {
            preprocessed_headers(&dir, &["root.h"]);
        });
        fs::remove_dir_all(&dir).ok();
        assert!(
            message.contains("UNKNOWN TYPE WORD") && message.contains(spelling),
            "`{spelling}` must fail the census by name rather than classify \
             as dtype-free: {message}"
        );
    }
}

/// Positive parity for the inverted rule: the spellings the published
/// headers actually use - qualifiers, aggregate keywords, the two
/// non-arithmetic value types, and the project's own opaque handles - pass
/// unremarked, and the numeric ones still classify.
#[test]
fn known_c_type_words_still_classify_without_rejection() {
    let dir = planted_include_dir(
        "known-type-words",
        &[(
            "root.h",
            "_Bool chelis_is_ready(const chelis_tensor *t);\n\
             void chelis_release(struct chelis_arena *arena);\n\
             extern char chelis_tag_byte;\n\
             double chelis_measure(unsigned long n);\n",
        )],
    );
    let per_file = preprocessed_headers(&dir, &["root.h"]);
    fs::remove_dir_all(&dir).ok();
    let rows: Vec<Row> = per_file
        .iter()
        .flat_map(|(name, text)| header_rows_local(name, text))
        .collect();
    let measure = rows
        .iter()
        .find(|row| row.id.contains("chelis_measure"))
        .expect("the numeric export is inventoried");
    assert!(
        measure.flags.iter().any(|flag| flag == "float-carrier"),
        "an accepted declaration is still classified: {measure:?}"
    );
    let ready = rows
        .iter()
        .find(|row| row.id.contains("chelis_is_ready"))
        .expect("the dtype-free export is inventoried");
    assert!(
        ready.flags.is_empty(),
        "`_Bool` and an opaque handle carry no dtype: {ready:?}"
    );
}

/// The typedef half of N1. A use site may answer "that word is a local
/// alias", which is only safe because the alias's own statement is checked
/// by the same rule - otherwise `typedef _Float16 chelis_half;` restores
/// the exact hole through a `chelis_`-prefixed name.
#[test]
fn a_typedef_alias_cannot_introduce_an_unknown_type_word() {
    let dir = planted_include_dir(
        "aliased-unknown-type-word",
        &[(
            "root.h",
            "typedef _Float16 chelis_half;\n\
             chelis_half chelis_halve(chelis_half value);\n",
        )],
    );
    let message = expect_census_panic(|| {
        preprocessed_headers(&dir, &["root.h"]);
    });
    fs::remove_dir_all(&dir).ok();
    assert!(
        message.contains("UNKNOWN TYPE WORD") && message.contains("_Float16"),
        "a project-named alias must not launder an unknown arithmetic \
         spelling: {message}"
    );
}

/// The positional read is what decides which words get classified, so a
/// declarator name shaped like a type spelling means the read was wrong.
/// Both reserved shapes are rejected; an ordinary parameter name is not.
#[test]
fn a_reserved_shaped_declarator_name_is_rejected() {
    for declarator in ["_reserved", "shape_t"] {
        let dir = planted_include_dir(
            "reserved-declarator",
            &[("root.h", &format!("void chelis_take(int {declarator});\n"))],
        );
        let message = expect_census_panic(|| {
            preprocessed_headers(&dir, &["root.h"]);
        });
        fs::remove_dir_all(&dir).ok();
        assert!(
            message.contains("RESERVED-SHAPED DECLARATOR NAME") && message.contains(declarator),
            "`{declarator}` must be rejected as a declarator: {message}"
        );
    }

    let dir = planted_include_dir(
        "ordinary-declarator",
        &[("root.h", "void chelis_take(int count);\n")],
    );
    preprocessed_headers(&dir, &["root.h"]);
    fs::remove_dir_all(&dir).ok();
}

/// N1's language-side companion. `NUMERIC_PRIMS`/`FLOAT_PRIMS` are string
/// lists because the desugared AST carries `(t-prim {} <name>)` symbols, so
/// nothing about widening `chelis_types::Prim` would otherwise reach this
/// file. The exhaustive match in `prim_census_class` is the compile-time
/// half of the lock; this is the agreement half, checked in both
/// directions.
#[test]
fn census_prim_lists_cover_every_prim_variant() {
    for prim in ALL_PRIMS {
        let name = prim.name();
        match prim_census_class(*prim) {
            PrimCensusClass::Float => {
                assert!(
                    FLOAT_PRIMS.contains(&name) && NUMERIC_PRIMS.contains(&name),
                    "float prim `{name}` must be enumerated as a seam carrier"
                );
            }
            PrimCensusClass::Integer => {
                assert!(
                    NUMERIC_PRIMS.contains(&name) && !FLOAT_PRIMS.contains(&name),
                    "integer prim `{name}` must be enumerated as `numeric-op`"
                );
            }
            PrimCensusClass::NonNumeric => {
                assert!(
                    !NUMERIC_PRIMS.contains(&name) && !FLOAT_PRIMS.contains(&name),
                    "dtype-free prim `{name}` must not be census surface"
                );
            }
        }
    }
    for name in NUMERIC_PRIMS {
        let prim = Prim::parse_name(name)
            .unwrap_or_else(|| panic!("`{name}` is not a chelis_types::Prim spelling"));
        assert_ne!(
            prim_census_class(prim),
            PrimCensusClass::NonNumeric,
            "`{name}` is enumerated as numeric but classified dtype-free"
        );
    }
    for name in FLOAT_PRIMS {
        let prim = Prim::parse_name(name).expect("float prim spelling");
        assert_eq!(
            prim_census_class(prim),
            PrimCensusClass::Float,
            "`{name}` is enumerated as a seam carrier but not classified float"
        );
    }
}

/// N2: the generic typedef word split popped the ARRAY EXTENT as the alias
/// name, so `typedef double chelis_vec4[4];` registered `4` and left
/// `chelis_vec4` unresolved. A setter taking it then inherited nothing, and
/// because the alias is `chelis_`-prefixed the inverted type-word rule
/// accepts it - so this is not subsumed by N1.
///
/// Reverting `collect_typedefs`' array branch turns this RED.
#[test]
fn an_array_typedef_cannot_launder_a_float_carrier() {
    let rows = header_rows_local(
        "planted.h",
        "typedef double chelis_vec4[4];\nvoid chelis_set(chelis_vec4 v);\n",
    );
    let setter = rows
        .iter()
        .find(|row| row.id.contains("chelis_set"))
        .expect("the setter is inventoried");
    assert!(
        setter.flags.iter().any(|flag| flag == "float-carrier"),
        "a setter taking an array-of-double alias carries the seam: {setter:?}"
    );

    let dtype_rows = header_rows_local(
        "planted.h",
        "typedef int chelis_dtype_pair[2];\nvoid chelis_route(chelis_dtype_pair dtype);\n",
    );
    let router = dtype_rows
        .iter()
        .find(|row| row.id.contains("chelis_route"))
        .expect("the router is inventoried");
    assert!(
        router.flags.iter().any(|flag| flag == "numeric-op"),
        "an array-of-int alias still resolves to its element type: {router:?}"
    );
}

/// Negative parity for the array branch, on the same footing the
/// parenthesized branch already has: a shape the resolver cannot read is
/// rejected rather than guessed at, because guessing is what produced the
/// laundering.
#[test]
fn an_unresolvable_array_typedef_is_rejected() {
    for stmt in ["typedef double [4];", "typedef chelis_vec4[4];"] {
        let message = expect_census_panic(move || {
            collect_typedefs(stmt);
        });
        assert!(
            message.contains("UNRESOLVABLE ARRAY TYPEDEF"),
            "`{stmt}` must fail loudly rather than register a bogus alias: {message}"
        );
    }
}

/// Parse one planted Surf module the way `stdlib_rows` does and run the
/// stdlib enumerator over it.
fn planted_stdlib_rows(label: &str, source: &str) -> Vec<Row> {
    let decls = chelis_surf::parser::parse_str(source).expect("planted module parses");
    let exprs = chelis_surf::desugar::desugar_program(&decls);
    let mut rows = Vec::new();
    scan_deftypes(&exprs, label, &mut rows);
    rows
}

/// N3: an exported `def` with no declared signature produced NO row and no
/// complaint. The enumerator reads capacity off the `defsig`, so such a def
/// is public numeric surface that is invisible and uncited at once - and
/// the Surf style guide recommends exactly that shape for load-style
/// top-level bindings, which puts the silent path one stdlib commit away.
///
/// Restoring the bare `continue` in `scan_exported_numeric_defs` turns this
/// RED.
#[test]
fn an_exported_stdlib_def_without_a_signature_fails_loudly() {
    let source = "module Std.Planted\n\
                  export (planted_weights)\n\
                  def planted_weights() = 1.0\n";
    let message = expect_census_panic(move || {
        planted_stdlib_rows("planted", source);
    });
    assert!(
        message.contains("EXPORTED DEFINITION WITHOUT A DECLARED SIGNATURE")
            && message.contains("planted::planted_weights"),
        "a signature-less exported def must name itself in the failure: {message}"
    );
}

/// Positive parity for N3, in both directions the enumerator cares about:
/// a declared signature enumerates (and classifies), and an export that
/// names no value definition at all - a type or ADT export - is not a
/// missing signature.
#[test]
fn a_declared_stdlib_signature_enumerates_and_a_type_export_does_not() {
    let rows = planted_stdlib_rows(
        "planted",
        "module Std.Planted\n\
         export (planted_weights)\n\
         sig planted_weights: unit -> f32\n\
         def planted_weights() = 1.0\n",
    );
    let row = rows
        .iter()
        .find(|row| row.kind == "std-def-numeric")
        .expect("a declared numeric signature enumerates");
    assert!(
        row.flags.iter().any(|flag| flag == "float-carrier"),
        "a bare f32 public value is a seam: {row:?}"
    );

    let type_only = planted_stdlib_rows(
        "planted",
        "module Std.Planted\n\
         export (PlantedTag)\n\
         type PlantedTag =\n\
         \x20 | PlantedOn\n\
         \x20 | PlantedOff\n",
    );
    assert!(
        type_only.is_empty(),
        "a dtype-free type export is not a signature-less def: {type_only:?}"
    );
}

/// chelis#1417 positive control: a `spec/04-type-system.md` §5.9 dtype-family
/// bound IS numeric capacity. The signature below writes no concrete dtype
/// and no metavariable spelling, so before bounds existed the domain lived
/// only in the binder's NAME. Deleting the bound-seeding loop in
/// `scan_exported_numeric_defs` turns this RED: the row vanishes and a
/// public numeric definition becomes invisible to the census.
#[test]
fn a_declared_dtype_family_bound_is_numeric_capacity() {
    for (family, expected) in [
        ("Float", "[p: Float]"),
        ("Int", "[p: Int]"),
        ("Numeric", "[p: Numeric]"),
    ] {
        let rows = planted_stdlib_rows(
            "planted",
            &format!(
                "module Std.Planted\n\
                 export (planted_pick)\n\
                 sig planted_pick[p: {family}]: p -> p -> p\n\
                 def planted_pick(a, b) = a\n"
            ),
        );
        let row = rows
            .iter()
            .find(|row| row.kind == "std-def-numeric")
            .unwrap_or_else(|| panic!("a `{family}`-bounded signature must enumerate: {rows:?}"));
        assert!(
            row.id.contains(expected),
            "the bound is part of the row identity, not just its capacity: {row:?}"
        );
        assert!(
            row.flags.is_empty(),
            "a declared domain is not a bare-numeric seam: {row:?}"
        );
    }
}

/// The negative half of the control: the same signature with the bound
/// removed carries no numeric capacity at all, so the bound is what makes
/// the row visible rather than some other spelling in the shape.
#[test]
fn the_same_signature_without_a_bound_is_not_numeric_capacity() {
    let rows = planted_stdlib_rows(
        "planted",
        "module Std.Planted\n\
         export (planted_pick)\n\
         sig planted_pick: p -> p -> p\n\
         def planted_pick(a, b) = a\n",
    );
    assert!(
        rows.iter().all(|row| row.kind != "std-def-numeric"),
        "an unbounded binder ranges over every type, not a dtype family: {rows:?}"
    );
}

/// Two signatures that differ only in their declared family are different
/// numeric surface. Dropping the bound from the identity string collapses
/// them into one row and lets a family change land uncited.
#[test]
fn signatures_differing_only_by_family_have_distinct_identities() {
    let identity = |family: &str| {
        planted_stdlib_rows(
            "planted",
            &format!(
                "module Std.Planted\n\
                 export (planted_pick)\n\
                 sig planted_pick[p: {family}]: p -> p -> p\n\
                 def planted_pick(a, b) = a\n"
            ),
        )
        .into_iter()
        .find(|row| row.kind == "std-def-numeric")
        .expect("bounded signature enumerates")
        .id
    };
    assert_ne!(identity("Float"), identity("Int"));
    assert_ne!(identity("Int"), identity("Numeric"));
}

/// N4, the missing control for the newest commit's claim. The macro-taint
/// component is built from the local include graph, and it must follow the
/// ANGLE spelling too: `cc -E -I` resolves `<x>` and `"x"` identically
/// inside the include directory. Here the include is UNCONDITIONAL and only
/// the `#define` is conditional, so no other guard fires - the multi-root
/// context check sees one root, the conditional-include check sees none,
/// and `root.h` declares nothing itself.
///
/// Narrowing `closure_conditional_macro_taint`'s edges to quoted includes
/// turns this RED and leaves the rest of the suite green, which is what
/// made the claim vacuous before.
#[test]
fn conditional_macro_taint_follows_angle_spelled_includes() {
    let dir = planted_include_dir(
        "angle-taint",
        &[
            (
                "root.h",
                "#ifdef CHELIS_REVIEW_WIDE\n\
                 #define CHELIS_NUM double\n\
                 #else\n\
                 #define CHELIS_NUM float\n\
                 #endif\n\
                 #include <shared.h>\n",
            ),
            (
                "shared.h",
                "CHELIS_NUM chelis_context_result(CHELIS_NUM value);\n",
            ),
        ],
    );
    let message = expect_census_panic(|| {
        preprocessed_headers(&dir, &["root.h"]);
    });
    fs::remove_dir_all(&dir).ok();
    assert!(
        message.contains("CONTEXT-VARYING PUBLIC ABI")
            && message.contains("shared.h")
            && message.contains("CHELIS_NUM"),
        "the taint component must cross an angle-spelled include and name \
         the tainted token: {message}"
    );
}

/// N5: the derived-roots rule compared the include closure against a FLAT
/// `read_dir`, so a subdirectory of the published include directory was an
/// uninventoried publishing channel - the same hole the rule exists to
/// close one level up.
///
/// Reverting `published_headers_on_disk` to a non-recursive walk turns the
/// first half RED.
#[test]
fn a_published_header_in_a_subdirectory_is_reached_or_fails() {
    let dir = planted_include_dir(
        "subdirectory-orphan",
        &[
            ("root.h", "void chelis_rooted(int x);\n"),
            (
                "detail/orphan.h",
                "double chelis_orphan(double value, int out_dtype);\n",
            ),
        ],
    );
    let message = expect_census_panic(|| {
        preprocessed_headers(&dir, &["root.h"]);
    });
    fs::remove_dir_all(&dir).ok();
    assert!(
        message.contains("PUBLISHED HEADER NOT REACHED FROM ANY ROOT")
            && message.contains("detail/orphan.h"),
        "a header one directory down is still published: {message}"
    );

    let reached = planted_include_dir(
        "subdirectory-reached",
        &[
            ("root.h", "#include \"detail/sub.h\"\n"),
            ("detail/sub.h", "void chelis_sub(int x);\n"),
        ],
    );
    let per_file = preprocessed_headers(&reached, &["root.h"]);
    fs::remove_dir_all(&reached).ok();
    assert!(
        per_file.values().any(|text| text.contains("chelis_sub")),
        "a subdirectory header a root includes is inventoried: {per_file:?}"
    );
}

/// N6: §C6 claims a new post-ratchet stdlib numeric callable owes the same
/// semantic registration as a new runtime callable, and the negative half had
/// no control - only the header-side `new_post_ratchet_runtime_numeric_op_requires_
/// semantic_registration` existed.
///
/// The second half covers the branch that binds `std-def-numeric` by KIND
/// rather than by flags. A float-only stdlib def under a maintainer
/// override reaches that branch and no other, so it is the case that
/// proves capacity disposition and callable semantics stay independent
/// obligations on this family. Dropping `|| row.kind ==
/// "std-def-numeric"` from the registration check turns it RED.
#[test]
fn a_new_stdlib_numeric_def_requires_semantic_registration() {
    let rows = planted_stdlib_rows(
        "planted",
        "module Std.Planted\n\
         export (planted_scale)\n\
         sig planted_scale: int32 -> int32\n\
         def planted_scale(n) = n\n",
    );
    let row = rows
        .into_iter()
        .find(|row| row.kind == "std-def-numeric")
        .expect("the exported numeric def enumerates");
    assert!(
        !is_seam(&row.flags),
        "an integer carrier is not a seam: {row:?}"
    );
    let baseline = Baseline {
        version: 3,
        legs: coverage_manifest(),
        rows: vec![row.clone()],
    };
    let err = check_against_baseline(std::slice::from_ref(&row), &baseline).unwrap_err();
    assert!(
        err.contains("UNCLASSIFIED census row") && err.contains("planted::planted_scale"),
        "an issue citation is not a semantic decision for the new stdlib \
         callable: {err}"
    );

    let registration = FinalNumericOperationRegistration {
        surface: StaticSurfaceDescriptor::new(
            PRIMARY_CENSUS_FAMILY,
            "std-def-numeric",
            "planted::planted_scale: (t-fn {} (t-prim {} int32) (t-prim {} int32))",
            &["numeric-op"],
        ),
        atom: "[05-OP-1]",
        authority_anchor: "Integer scale",
    };
    assert!(
        check_against_baseline_with_authorities(
            std::slice::from_ref(&row),
            &baseline,
            &[],
            "Synthetic controlling fixture:\n> **[05-OP-1]** Integer scale.",
            AuthorityRegistries {
                nonnumeric: &[],
                tagged_transports: &[],
                numeric_operations: &[registration],
            },
        )
        .is_ok(),
        "an exact registration against a real atom is the sanctioned path"
    );

    // The KIND branch. A float-only def carries `float-carrier` and NOT
    // `numeric-op`, so the flags branch above cannot reach it; a maintainer
    // override is the only disposition that gets such a row past the seam
    // rule, and it must still not waive the semantics obligation.
    let float_rows = planted_stdlib_rows(
        "planted",
        "module Std.Planted\n\
         export (planted_ratio)\n\
         sig planted_ratio: unit -> f32\n\
         def planted_ratio() = 1.0\n",
    );
    let mut float_row = float_rows
        .into_iter()
        .find(|row| row.kind == "std-def-numeric")
        .expect("the exported float def enumerates");
    assert_eq!(
        float_row.flags,
        vec!["float-carrier".to_string()],
        "a float-only public def is a seam and nothing else: {float_row:?}"
    );
    float_row.citation = "maintainer-override(FFI staging, chelis#893)".to_string();
    let float_baseline = Baseline {
        version: 3,
        legs: coverage_manifest(),
        rows: vec![float_row.clone()],
    };
    let float_err =
        check_against_baseline(std::slice::from_ref(&float_row), &float_baseline).unwrap_err();
    assert!(
        float_err.contains("UNCLASSIFIED census row")
            && float_err.contains("planted::planted_ratio"),
        "a capacity override is not final authority for the new stdlib \
         callable: {float_err}"
    );
}

/// The stdlib mirror of the "an unrelated atom is not authority" rule,
/// which the header family already covers: registering the callable
/// against an atom that does not exist in the controlling spec fails on
/// the registration, not on the citation.
#[test]
fn a_stdlib_registration_against_a_nonexistent_atom_fails() {
    let rows = planted_stdlib_rows(
        "planted",
        "module Std.Planted\n\
         export (planted_scale)\n\
         sig planted_scale: int32 -> int32\n\
         def planted_scale(n) = n\n",
    );
    let mut row = rows
        .into_iter()
        .find(|row| row.kind == "std-def-numeric")
        .expect("the exported numeric def enumerates");
    row.citation = "chelis#729".to_string();
    let baseline = Baseline {
        version: 3,
        legs: coverage_manifest(),
        rows: vec![row.clone()],
    };
    let registration = SemanticRegistration {
        callable: Box::leak(callable_identity(&row).into_boxed_str()),
        atom: "[05-OP-999]",
    };
    let err = check_against_baseline_with(
        std::slice::from_ref(&row),
        &baseline,
        &[registration],
        "Synthetic controlling fixture:\n> **[05-OP-1]** Integer scale.",
    )
    .unwrap_err();
    assert!(
        err.contains("does not exist as a normative"),
        "naming an absent atom is not registration: {err}"
    );
}

/// The `#line` ban reads DIRECTIVES, and phase-2 line splicing joins
/// `#\<newline>line ...` into one before any directive is recognized. The
/// physical-line scan missed it and the independent attribution backstop
/// caught the export instead - the right outcome under the wrong name.
/// This is a diagnostic-quality closure, not a hole: the export never
/// entered the inventory either way.
#[test]
fn a_spliced_line_directive_is_caught_by_the_ban_itself() {
    let dir = planted_include_dir(
        "spliced-line-directive",
        &[(
            "root.h",
            "#\\\nline 1 \"/opt/vendor/x.h\"\nvoid chelis_smuggled(int x);\n",
        )],
    );
    let message = expect_census_panic(|| {
        preprocessed_headers(&dir, &["root.h"]);
    });
    fs::remove_dir_all(&dir).ok();
    assert!(
        message.contains("LINE-DIRECTIVE SPOOFING SURFACE"),
        "a spliced directive is the same directive: {message}"
    );
}

/// prelude-adt-numeric negative control: a Rust-registered prelude ADT
/// with a bare-f64 variant is detected as a float-carrier seam by the
/// leg's own classifier -- the chelis#891 JNum shape at the Rust registry,
/// mirroring `planted_deftype_with_f64_variant_is_detected` on the `.ch`
/// leg. Built through the same `AdtDef` value the registry stores, so
/// the control exercises the exact enumeration path minus only the
/// `register_prelude_adts` source.
#[test]
fn planted_prelude_adt_with_f64_variant_is_detected() {
    use chelis_types::adt::{AdtDef, VariantInfo};
    use chelis_types::types::Type;
    let planted = AdtDef {
        name: "Planted".to_string(),
        type_params: Vec::new(),
        param_kinds: Vec::new(),
        param_vars: Vec::new(),
        param_args: Vec::new(),
        opaque: false,
        defining_module: None,
        variants: vec![VariantInfo {
            name: "PFloat".to_string(),
            fields: vec![(None, Type::Prim(Prim::F64))],
        }],
    };
    let mut prims = BTreeSet::new();
    for variant in &planted.variants {
        for (_, ty) in &variant.fields {
            collect_prelude_numeric_prims(ty, &mut prims);
        }
    }
    let flags = numeric_carrier_flags(&prims);
    assert_eq!(flags, vec!["float-carrier".to_string()], "{prims:?}");
    assert!(is_seam(&flags), "a bare-f64 prelude variant is a seam");
    // And the renderer preserves the field shape in the identity.
    assert_eq!(
        render_prelude_census_type(&Type::Prim(Prim::F64)),
        "f64",
        "identity must spell the dtype"
    );
}

/// chelis#759 / MEDIUM-2: pin the PRESENCE of `cast_trunc`'s semantic
/// registration.
///
/// `registration_problem` only validates the atom a row points AT; it
/// never asks whether a given callable has a row, because `cast_trunc`
/// is compiler-owned and no census enumerator produces one for it. That
/// left the entry deletable with the whole suite still green. This test
/// is the missing direction: removing the entry, or repointing it at a
/// different atom, goes red here.
#[test]
fn cast_trunc_is_registered_against_its_authority_atom() {
    const CAST_TRUNC: &str = "[compiler-builtin-numeric] cast_trunc(source: f16 | bf16 | f32 | f64, \
         target: int8 | int16 | int32 | int64) -> int8 | int16 | int32 | int64";
    let registration = SEMANTIC_REGISTRATIONS
        .iter()
        .find(|r| r.callable == CAST_TRUNC)
        .unwrap_or_else(|| {
            panic!(
                "the `cast_trunc` semantic registration is missing. A new numeric \
                 op requires an exact registration bound to one verbatim [05-OP-N] \
                 atom in the same change set (AGENTS.md section Numeric Surface \
                 Discipline; spec/design/dtype_semantics.md section C6). Expected \
                 callable identity:\n  {CAST_TRUNC}"
            )
        });
    assert_eq!(
        registration.atom, "[05-OP-6]",
        "`cast_trunc`'s controlling atom is spec/05-risc-primitives.md section 3.8 \
         [05-OP-6]; no other atom's normative text governs a truncating cast"
    );
    // And the atom it names really exists as a normative definition, so
    // this test cannot pass against a dangling reference.
    let spec = fs::read_to_string(repo_root().join(CONTROLLING_SPEC_REL))
        .expect("controlling spec/05 must be readable");
    assert!(
        registration_problem(*registration, &spec).is_none(),
        "the `cast_trunc` registration must satisfy the same atom-existence \
         contract as every other registered callable"
    );
}

#[test]
fn fixed_control_dropout_is_registered_against_its_exact_authority_atom() {
    let callable = "[compiler-builtin-numeric] dropout(input: &tensor[D, p_float], rate: p_float) -> tensor[D, p_float]";
    let registration = SEMANTIC_REGISTRATIONS
        .iter()
        .find(|row| row.callable == callable)
        .expect("the active-float dropout signature requires an exact semantic registration");
    assert_eq!(registration.atom, "[05-OP-37]");
    let spec = fs::read_to_string(repo_root().join(CONTROLLING_SPEC_REL)).unwrap();
    assert!(registration_problem(*registration, &spec).is_none());
}

#[test]
fn prepared_dropout_kernel_boundaries_have_exact_semantic_authority() {
    let spec = fs::read_to_string(repo_root().join(CONTROLLING_SPEC_REL)).unwrap();
    for callable in [
        "[compiler-kernel-numeric] chelis_types::dtype_semantics::PreparedDropout::new(input: &TensorStorage, rate: ScalarValue) -> Result<PreparedDropout, NumericKernelError>",
        "[compiler-kernel-numeric] chelis_types::dtype_semantics::PreparedDropout::apply(&self, seed: u64, ordinal: u64) -> Result<TensorStorage, NumericKernelError>",
    ] {
        let rows = SEMANTIC_REGISTRATIONS
            .iter()
            .filter(|row| row.callable == callable)
            .collect::<Vec<_>>();
        assert_eq!(
            rows.len(),
            1,
            "missing or duplicate kernel boundary: {callable}"
        );
        assert_eq!(
            rows[0].atom, "[05-OP-37]",
            "wrong kernel authority: {callable}"
        );
        assert!(registration_problem(*rows[0], &spec).is_none());
    }
}

#[test]
fn post_1167_compiler_numeric_builtins_have_exact_authority_registrations() {
    let expected = [
        (
            "[compiler-builtin-numeric] uniform_like(template: &tensor[D, p], low: f32, high: f32) -> tensor[D, p]",
            "[05-OP-8]",
        ),
        (
            "[compiler-builtin-numeric] pad_sequences(sequences: List[List[T]], pad: T) -> tensor[len(sequences), width, T]",
            "[05-OP-9]",
        ),
        (
            "[compiler-builtin-numeric] pad_sequences_to(sequences: List[List[T]], width: int64, pad: T) -> tensor[len(sequences), width, T]",
            "[05-OP-10]",
        ),
    ];
    let spec = fs::read_to_string(repo_root().join(CONTROLLING_SPEC_REL))
        .expect("controlling spec/05 must be readable");
    for (callable, atom) in expected {
        let registration = SEMANTIC_REGISTRATIONS
            .iter()
            .find(|registration| registration.callable == callable)
            .unwrap_or_else(|| panic!("missing semantic registration for `{callable}`"));
        assert_eq!(registration.atom, atom, "wrong authority for `{callable}`");
        assert!(
            registration_problem(*registration, &spec).is_none(),
            "`{callable}` must name an existing normative atom"
        );
    }
}

#[test]
fn count_is_registered_against_its_exact_authority_atom() {
    const COUNT: &str = "[compiler-builtin-numeric] count(input: &tensor[D, bool], axes: int32...) \
         -> tensor[D\\axes, int64]";
    let registration = SEMANTIC_REGISTRATIONS
        .iter()
        .find(|registration| registration.callable == COUNT)
        .unwrap_or_else(|| panic!("missing semantic registration for `{COUNT}`"));
    assert_eq!(registration.atom, "[05-OP-29]");
    let spec = fs::read_to_string(repo_root().join(CONTROLLING_SPEC_REL))
        .expect("controlling spec/05 must be readable");
    assert!(
        registration_problem(*registration, &spec).is_none(),
        "count must name the existing [05-OP-29] normative atom"
    );
}

/// chelis#1313: the structural ReLU identity and its dedicated adjoint are
/// compiler-owned numeric operations. Neither is discoverable through a
/// published C header, stdlib def, or prelude ADT, so pin both directions of
/// their exact [05-OP-43] registration here.
#[test]
fn relu_identities_are_registered_against_their_exact_authority_atom() {
    let expected = [
        (
            "[compiler-builtin-numeric] relu(input: &tensor[D, p]) -> tensor[D, p]",
            "[05-OP-43]",
        ),
        (
            "[compiler-risc-numeric] relu_adjoint(input: &tensor[D, p], cotangent: &tensor[D, p]) -> tensor[D, p]",
            "[05-OP-43]",
        ),
    ];
    let spec = fs::read_to_string(repo_root().join(CONTROLLING_SPEC_REL))
        .expect("controlling spec/05 must be readable");
    for (callable, atom) in expected {
        let registration = SEMANTIC_REGISTRATIONS
            .iter()
            .find(|registration| registration.callable == callable)
            .unwrap_or_else(|| panic!("missing semantic registration for `{callable}`"));
        assert_eq!(registration.atom, atom, "wrong authority for `{callable}`");
        assert!(
            registration_problem(*registration, &spec).is_none(),
            "`{callable}` must name the existing [05-OP-43] normative atom"
        );
    }
}

#[test]
fn metadata_plan_callables_require_exact_op33_authority() {
    let root = repo_root();
    let files = preprocessed_headers(&root.join(INCLUDE_DIR_REL), HEADER_ROOTS);
    let mut typedefs = BTreeMap::new();
    for text in files.values() {
        typedefs.append(&mut collect_typedefs(text));
    }
    let rows: Vec<_> = files
        .iter()
        .flat_map(|(name, text)| header_rows(name, text, &typedefs))
        .filter(|row| row.kind == "header-export" && row.id.contains("chelis_metadata_plan_"))
        .collect();
    assert_eq!(
        rows.len(),
        11,
        "complete checked metadata plan API: {rows:?}"
    );
    let spec = fs::read_to_string(root.join(CONTROLLING_SPEC_REL)).unwrap();
    let registry = fs::read_to_string(root.join("spec/registry/c_tensor_runtime.md")).unwrap();
    let normative: BTreeSet<_> = registry
        .lines()
        .filter(|line| line.contains("chelis_metadata_plan_"))
        .map(|line| {
            let signature = line.split('`').nth(1).expect("exact normative C signature");
            format!(
                "chelis_runtime.h: {}",
                canonical_c_tokens(&format!("{signature};"))
            )
        })
        .collect();
    assert_eq!(
        rows.iter()
            .map(|row| row.id.clone())
            .collect::<BTreeSet<_>>(),
        normative
    );
    for row in rows {
        let surface = authority_surface(&row);
        assert_eq!(
            capacity_census_authority::classify_final_authority(
                &surface,
                final_authority_registries(),
                &spec,
            ),
            Ok(capacity_census_authority::FinalAuthority::NumericOperation { atom: "[05-OP-33]" }),
        );
        let mut successor = surface;
        successor.id = successor
            .id
            .replace("chelis_metadata_plan_", "chelis_unchecked_metadata_");
        assert!(
            capacity_census_authority::classify_final_authority(
                &successor,
                final_authority_registries(),
                &spec,
            )
            .is_err(),
            "a renamed successor must acquire independent exact authority"
        );
    }
}

#[test]
fn final_surface_has_no_duplicate_prelude_json() {
    assert!(
        prelude_adt_rows().is_empty(),
        "the final language has no prelude numeric ADT: {:?}",
        prelude_adt_rows()
    );
}

fn normative_registry_identities(registry: &str) -> BTreeSet<&str> {
    registry
        .lines()
        .filter_map(|line| line.strip_prefix("| `"))
        .filter_map(|rest| rest.split_once('`').map(|(identity, _)| identity))
        .collect()
}

#[test]
fn final_stdlib_registrations_are_exact_and_bijective_with_normative_registries() {
    let current_rows = stdlib_rows(&repo_root());
    let current: BTreeSet<(&str, &str, Vec<&str>)> = current_rows
        .iter()
        .map(|row| {
            (
                row.kind.as_str(),
                row.id.as_str(),
                row.flags.iter().map(String::as_str).collect(),
            )
        })
        .collect();
    let registered: BTreeSet<(&str, &str, Vec<&str>)> = FINAL_NUMERIC_OPERATION_ROWS
        .iter()
        .filter(|registration| registration.surface.kind.starts_with("std-"))
        .map(|registration| {
            (
                registration.surface.kind,
                registration.surface.id,
                registration.surface.flags.to_vec(),
            )
        })
        .collect();
    assert_eq!(
        current, registered,
        "every final stdlib capacity descriptor must have one exact final numeric-operation registration"
    );

    for (kind, registry, expected_atom) in [
        (
            "std-adt-numeric",
            include_str!("../../../spec/registry/stdlib_adt_identities.md"),
            "[05-OP-34]",
        ),
        (
            "std-def-numeric",
            include_str!("../../../spec/registry/stdlib_numeric_manifest.md"),
            "[05-OP-35]",
        ),
    ] {
        let registered_identities: BTreeSet<&str> = FINAL_NUMERIC_OPERATION_ROWS
            .iter()
            .filter(|registration| registration.surface.kind == kind)
            .map(|registration| {
                assert_eq!(registration.atom, expected_atom);
                registration
                    .surface
                    .id
                    .split_once(": ")
                    .expect("complete stdlib descriptor includes its signature")
                    .0
            })
            .collect();
        assert_eq!(
            registered_identities,
            normative_registry_identities(registry),
            "final {kind} registrations must be bijective with their normative registry"
        );
    }
}
