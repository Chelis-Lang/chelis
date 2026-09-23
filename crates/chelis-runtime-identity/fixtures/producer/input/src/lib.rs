include!(concat!(env!("OUT_DIR"), "/value.rs"));
#[cfg(feature = "boosted")]
pub const VALUE: u32 = fixture_runtime_leaf::LEAF + GENERATED + 100;
#[cfg(not(feature = "boosted"))]
pub const VALUE: u32 = fixture_runtime_leaf::LEAF + GENERATED;
