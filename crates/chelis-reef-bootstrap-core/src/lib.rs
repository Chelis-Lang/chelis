//! Allocator-backed bootstrap graph policy without host dependencies.
//!
//! Identity text is exact. Each input index is a separate occurrence, even when
//! identities repeat. Edges select the last matching occurrence. The checked
//! entry point reports conflicts before missing edges, then detects cycles.
//!
//! This crate has no dependencies, feature flags, or other core prerequisite.
//! Both default feature modes expose the same allocating API.
//!
//! The caller supplies ordered values and an allocator. Reef retains manifest
//! fetch, runtime filtering, release selection, and installation. Reef converts
//! typed errors to its existing diagnostic text without a second graph policy.
//! Target library compilation does not establish allocator capacity or device execution.
//!
//! ```
//! use chelis_reef_bootstrap_core::{validate_and_order, BootstrapIdentity, BootstrapNode};
//! let identity = BootstrapIdentity { name: "shell".into(), version: "1".into() };
//! let nodes = [BootstrapNode { identity, dependencies: vec![] }];
//! assert_eq!(validate_and_order(&nodes), Ok(vec![0]));
//! ```

#![no_std]

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

/// Exact package name and version text without version normalization.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BootstrapIdentity {
    pub name: String,
    pub version: String,
}

/// One input occurrence with dependencies in traversal order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BootstrapNode {
    pub identity: BootstrapIdentity,
    pub dependencies: Vec<BootstrapIdentity>,
}

/// Checked graph failures with structured identity evidence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BootstrapGraphError {
    /// One name has multiple distinct, lexically sorted version strings.
    DuplicateVersion {
        package: String,
        versions: Vec<String>,
    },
    /// The first unresolved edge in input node and dependency order.
    MissingDependency {
        dependent: BootstrapIdentity,
        missing: BootstrapIdentity,
    },
    /// The active identity suffix, with its target repeated to close the path.
    Cycle { cycle: Vec<BootstrapIdentity> },
}

/// Validate the complete graph and return dependency-first input indices.
///
/// Version conflicts precede missing edges. All exact edges must resolve before
/// iterative depth-first traversal starts. Each successful result contains every
/// input occurrence once. Repeated edges do not repeat finished occurrences.
///
/// Independent roots follow input order. Supplied edge order controls siblings.
/// This function does not promise a permutation-independent topological order.
/// When several names conflict, no particular conflict name is guaranteed.
pub fn validate_and_order(nodes: &[BootstrapNode]) -> Result<Vec<usize>, BootstrapGraphError> {
    let mut index_by_key: BTreeMap<(&str, &str), usize> = BTreeMap::new();
    for (index, node) in nodes.iter().enumerate() {
        index_by_key.insert(
            (node.identity.name.as_str(), node.identity.version.as_str()),
            index,
        );
    }
    // Distinct keys already supply sorted unique versions without another table.
    let mut prior_name = None;
    for &(name, _) in index_by_key.keys() {
        if prior_name == Some(name) {
            let versions = index_by_key
                .range((name, "")..)
                .map(|(&(candidate, version), _)| (candidate, version))
                .take_while(|(candidate, _)| *candidate == name)
                .map(|(_, version)| String::from(version))
                .collect();
            return Err(BootstrapGraphError::DuplicateVersion {
                package: name.into(),
                versions,
            });
        }
        prior_name = Some(name);
    }
    for node in nodes {
        for dependency in &node.dependencies {
            if !index_by_key.contains_key(&(dependency.name.as_str(), dependency.version.as_str()))
            {
                return Err(BootstrapGraphError::MissingDependency {
                    dependent: node.identity.clone(),
                    missing: dependency.clone(),
                });
            }
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Color {
        White,
        Gray,
        Black,
    }

    let mut colors = vec![Color::White; nodes.len()];
    let mut order = Vec::with_capacity(nodes.len());
    // Frames also form the active path. Identity text remains borrowed until an error.
    let mut path: Vec<(usize, usize)> = Vec::new();
    for start in 0..nodes.len() {
        if colors[start] != Color::White {
            continue;
        }
        colors[start] = Color::Gray;
        path.push((start, 0));
        while let Some(&(index, cursor)) = path.last() {
            let dependencies = &nodes[index].dependencies;
            if cursor == dependencies.len() {
                colors[index] = Color::Black;
                order.push(index);
                path.pop();
                continue;
            }
            path.last_mut().expect("active frame exists").1 += 1;
            let dependency = &dependencies[cursor];
            let target = index_by_key[&(dependency.name.as_str(), dependency.version.as_str())];
            match colors[target] {
                Color::Black => {}
                Color::White => {
                    colors[target] = Color::Gray;
                    path.push((target, 0));
                }
                Color::Gray => {
                    let first = path
                        .iter()
                        .position(|&(active, _)| nodes[active].identity == *dependency)
                        .expect("gray identity is on the active path");
                    let mut cycle = Vec::with_capacity(path.len() - first + 1);
                    cycle.extend(
                        path[first..]
                            .iter()
                            .map(|&(active, _)| nodes[active].identity.clone()),
                    );
                    cycle.push(dependency.clone());
                    return Err(BootstrapGraphError::Cycle { cycle });
                }
            }
        }
    }
    Ok(order)
}
