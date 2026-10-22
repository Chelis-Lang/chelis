/// The named lossy rungs of the chelis#759 cast ladder. Each one is a
/// separate atom; the lowering carries the rung on one `NamedCast` op so
/// every backend, evaluator, and adjoint site states each rung's
/// disposition by exhaustive matching instead of inheriting the checked
/// default's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NamedCastMode {
    /// `cast_trunc`, [05-OP-6]: float to integer, truncated toward zero;
    /// traps `Overflow` out of range and `Domain` on a non-finite source.
    Trunc,
    /// `cast_saturate`, [05-OP-23]: signed integer or float to integer,
    /// truncated toward zero and clamped to the target range; traps
    /// `Domain` on NaN and never `Overflow`.
    Saturate,
    /// `cast_wrap`, [05-OP-24]: signed integer to integer, the target-width
    /// two's complement representative; never traps.
    Wrap,
}

impl NamedCastMode {
    /// Every named rung, in ladder order.
    pub const ALL: &'static [NamedCastMode] = &[
        NamedCastMode::Trunc,
        NamedCastMode::Saturate,
        NamedCastMode::Wrap,
    ];

    /// The Surf keyword and the operation name traps and diagnostics carry.
    pub fn keyword(self) -> &'static str {
        match self {
            NamedCastMode::Trunc => "cast_trunc",
            NamedCastMode::Saturate => "cast_saturate",
            NamedCastMode::Wrap => "cast_wrap",
        }
    }

    /// The Deep mode selector symbol.
    pub fn deep_selector(self) -> &'static str {
        match self {
            NamedCastMode::Trunc => "trunc",
            NamedCastMode::Saturate => "saturate",
            NamedCastMode::Wrap => "wrap",
        }
    }

    /// The governing [05-OP-N] atom.
    pub fn atom(self) -> &'static str {
        match self {
            NamedCastMode::Trunc => "[05-OP-6]",
            NamedCastMode::Saturate => "[05-OP-23]",
            NamedCastMode::Wrap => "[05-OP-24]",
        }
    }
}
