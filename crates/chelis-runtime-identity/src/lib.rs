//! Data-only runtime identity derivation and retained native record decoding.
//! Callers observe and read inputs; this crate owns canonical policy and errors.

mod derive;
mod input;
mod model;
mod record;

pub use derive::{compare, derive_descriptor};
pub use input::{
    UNSELECTED_COMPONENTS, decode_provenance, encode_provenance, normalize_paths, plan_inputs,
};
pub use model::*;
pub use record::{
    RECORD_LENGTH, decode_archive, decode_archive_provenance, decode_image,
    decode_image_provenance, decode_record, encode_record,
};

#[cfg(test)]
mod tests;
