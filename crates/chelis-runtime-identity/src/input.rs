use crate::*;
use std::collections::{BTreeMap, BTreeSet};

pub(crate) fn validate_path(path: &str) -> Result<(), InputError> {
    if path.is_empty()
        || path.contains(['\\', ':', '\0'])
        || path.chars().any(char::is_control)
        || path
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
    {
        return Err(InputError::InvalidPath { path: path.into() });
    }
    Ok(())
}

fn selected(path: &str) -> bool {
    if path.split('/').any(|part| {
        matches!(
            part,
            ".git"
                | "target"
                | ".venv"
                | ".devenv"
                | "node_modules"
                | "tests"
                | "test"
                | "benches"
                | "bench"
                | "examples"
                | "example"
        )
    }) {
        return false;
    }
    let name = path.rsplit('/').next().unwrap_or(path);
    if matches!(
        name,
        "Cargo.toml"
            | "Cargo.lock"
            | "rust-toolchain"
            | "rust-toolchain.toml"
            | "build.rs"
            | "build.ninja"
            | "Makefile"
            | "CMakeLists.txt"
            | "config.toml"
    ) {
        return true;
    }
    let extension = name.rsplit_once('.').map(|(_, ext)| ext).unwrap_or("");
    matches!(
        extension,
        "rs" | "c"
            | "cc"
            | "cpp"
            | "cxx"
            | "h"
            | "hh"
            | "hpp"
            | "hxx"
            | "inc"
            | "inl"
            | "S"
            | "s"
            | "asm"
            | "m"
            | "mm"
            | "cu"
            | "cuh"
            | "hip"
            | "metal"
            | "ld"
            | "lds"
            | "cmake"
            | "nix"
    )
}

fn classify(path: &str, fallback: InputClass) -> InputClass {
    match path.rsplit_once('.').map(|(_, ext)| ext) {
        Some("h" | "hh" | "hpp" | "hxx" | "inc" | "inl" | "cuh") => InputClass::Header,
        _ if path.ends_with("Cargo.toml")
            || path.ends_with("Cargo.lock")
            || path.ends_with("build.rs") =>
        {
            InputClass::Build
        }
        _ => fallback,
    }
}

pub fn plan_inputs(roots: &[InventoryRoot]) -> Result<Vec<RequiredInput>, InputError> {
    let mut planned = BTreeMap::new();
    for root in roots {
        if !root.logical_prefix.is_empty() {
            validate_path(&root.logical_prefix)?;
        }
        let mut available = BTreeSet::new();
        for path in &root.files {
            validate_path(path)?;
            if !available.insert(path.as_str()) {
                return Err(InputError::DuplicateInput { path: path.clone() });
            }
        }
        let mut required = BTreeSet::new();
        for path in &root.explicitly_required {
            validate_path(path)?;
            if !required.insert(path.as_str()) {
                return Err(InputError::DuplicateInput { path: path.clone() });
            }
            if !available.contains(path.as_str()) {
                return Err(InputError::MissingInput { path: path.clone() });
            }
        }
        for path in &root.files {
            if !required.contains(path.as_str()) && !selected(path) {
                continue;
            }
            let logical = if root.logical_prefix.is_empty() {
                path.clone()
            } else {
                format!("{}/{path}", root.logical_prefix)
            };
            let class = classify(path, root.class);
            if planned.insert(logical.clone(), class).is_some() {
                return Err(InputError::DuplicateInput { path: logical });
            }
        }
    }
    Ok(planned
        .into_iter()
        .map(|(logical_path, class)| RequiredInput {
            logical_path,
            class,
        })
        .collect())
}

pub(crate) fn validate_mappings(mappings: &[PathMapping]) -> Result<(), InputError> {
    let mut physical = BTreeSet::new();
    for mapping in mappings {
        validate_path(&mapping.logical)?;
        if !mapping.physical.starts_with('/')
            || mapping.physical.ends_with('/')
            || mapping.physical.contains(['\\', '\0'])
            || mapping.physical.chars().any(char::is_control)
            || mapping.physical[1..]
                .split('/')
                .any(|part| part.is_empty() || part == "." || part == "..")
            || !physical.insert(&mapping.physical)
        {
            return Err(InputError::InvalidMapping {
                physical: mapping.physical.clone(),
            });
        }
    }
    Ok(())
}

pub fn normalize_paths(value: &str, mappings: &[PathMapping]) -> Result<String, InputError> {
    validate_mappings(mappings)?;
    let mut mappings: Vec<_> = mappings.iter().collect();
    mappings.sort_by(|a, b| {
        b.physical
            .len()
            .cmp(&a.physical.len())
            .then_with(|| a.physical.cmp(&b.physical))
    });
    let mut output = String::with_capacity(value.len());
    let mut offset = 0;
    while offset < value.len() {
        let suffix = &value[offset..];
        let before = value[..offset].chars().next_back();
        let boundary = before
            .is_none_or(|ch| !ch.is_alphanumeric() && !matches!(ch, '/' | '_' | '-' | '.' | '\\'));
        let mapping = if boundary {
            mappings.iter().find(|mapping| {
                if !suffix.starts_with(&mapping.physical) {
                    return false;
                }
                suffix[mapping.physical.len()..]
                    .chars()
                    .next()
                    .is_none_or(|ch| !ch.is_alphanumeric() && !matches!(ch, '_' | '-' | '.' | '\\'))
            })
        } else {
            None
        };
        if let Some(mapping) = mapping {
            output.push_str(&mapping.logical);
            offset += mapping.physical.len();
        } else if let Some(ch) = suffix.chars().next() {
            output.push(ch);
            offset += ch.len_utf8();
        }
    }
    Ok(output)
}

pub fn encode_provenance(provenance: &BuildProvenance) -> Result<Vec<u8>, InputError> {
    validate_provenance(provenance)?;
    serde_json::to_vec(provenance).map_err(|error| InputError::Serialization {
        message: error.to_string(),
    })
}

pub fn decode_provenance(bytes: &[u8]) -> Result<BuildProvenance, InputError> {
    let provenance = serde_json::from_slice(bytes).map_err(|error| InputError::Serialization {
        message: error.to_string(),
    })?;
    validate_provenance(&provenance)?;
    Ok(provenance)
}

fn validate_provenance(provenance: &BuildProvenance) -> Result<(), InputError> {
    if let BuildProvenance::SourceWorktree {
        source_root,
        recipe,
        roots,
    } = provenance
    {
        validate_mappings(roots)?;
        if source_root.is_empty() || !roots.iter().any(|mapping| &mapping.physical == source_root) {
            return Err(InputError::InvalidObservation {
                field: "source provenance root mapping".into(),
            });
        }
        crate::derive::validate_recipe(recipe)?;
    }
    Ok(())
}
