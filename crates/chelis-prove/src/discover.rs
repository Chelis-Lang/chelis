//! Property discovery from Surf AST.
//!
//! Scans parsed Surf declarations for `Decl::Property` nodes and returns
//! structured property descriptors for downstream verification tiers.

use chelis_surf::ast::{Decl, Expr, Literal, Param, PropertyOption};

/// A property discovered from Surf source.
#[derive(Debug, Clone)]
pub struct DiscoveredProperty {
    pub name: String,
    pub params: Vec<Param>,
    pub preconditions: Vec<Expr>,
    pub body: Expr,
    pub samples: Option<usize>,
    pub seed: Option<u64>,
}

/// Collect all `@property` declarations from parsed Surf decls.
///
/// If `only` is provided, filters by name (supports trailing `*` glob and
/// substring match).
pub fn collect_surf_properties(decls: &[Decl], only: Option<&str>) -> Vec<DiscoveredProperty> {
    decls
        .iter()
        .filter_map(|decl| match decl {
            Decl::Property {
                name,
                params,
                preconditions,
                body,
                options,
                ..
            } if matches_filter(name, only) => Some(DiscoveredProperty {
                name: name.clone(),
                params: params.clone(),
                preconditions: preconditions.clone(),
                body: body.clone(),
                samples: property_samples(options),
                seed: property_seed(options),
            }),
            _ => None,
        })
        .collect()
}

/// Extract the `samples` option from property options.
pub fn property_samples(options: &[PropertyOption]) -> Option<usize> {
    options.iter().find_map(|option| match option {
        PropertyOption::Samples(Expr::Lit(Literal::Int(value), _), _) => {
            (*value >= 0).then_some(*value as usize)
        }
        _ => None,
    })
}

/// Extract the `seed` option from property options.
pub fn property_seed(options: &[PropertyOption]) -> Option<u64> {
    options.iter().find_map(|option| match option {
        PropertyOption::Seed(Expr::Lit(Literal::Int(value), _), _) => {
            (*value >= 0).then_some(*value as u64)
        }
        _ => None,
    })
}

fn matches_filter(name: &str, only: Option<&str>) -> bool {
    let Some(pattern) = only else {
        return true;
    };
    if pattern == name {
        return true;
    }
    if let Some(prefix) = pattern.strip_suffix('*') {
        return name.starts_with(prefix);
    }
    name.contains(pattern)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_filter_no_pattern() {
        assert!(matches_filter("anything", None));
    }

    #[test]
    fn matches_filter_exact() {
        assert!(matches_filter("foo", Some("foo")));
        assert!(!matches_filter("bar", Some("foo")));
    }

    #[test]
    fn matches_filter_glob() {
        assert!(matches_filter("foo_bar", Some("foo*")));
        assert!(!matches_filter("baz_bar", Some("foo*")));
    }

    #[test]
    fn matches_filter_substring() {
        assert!(matches_filter("test_foo_bar", Some("foo")));
        assert!(!matches_filter("test_baz_bar", Some("foo")));
    }
}
