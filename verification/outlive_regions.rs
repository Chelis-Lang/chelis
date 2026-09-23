//! Input declarations for the direct OCaml outlive-region smoke check.
//! These deliberately separate declared constraints from reference nesting.

pub struct RegionOutlives<'short, 'long: 'short, 'unrelated> {
    pub short: &'short mut u32,
    pub long: &'long mut u32,
    pub unrelated: &'unrelated mut u32,
}

pub struct TypeOutlives<'short, T: 'short> {
    pub value: &'short T,
}
