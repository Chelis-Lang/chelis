//! Closed key primitive and admission identities for [04-LIN-9].

/// An operation whose own atom names `key` as its first operand, which it
/// consumes: the key derivations [05-OP-70] to [05-OP-72] and the draws
/// [05-OP-8] and [05-OP-37]. [05-OP-69] creates a key from a seed and
/// takes none.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KeyPrimitive {
    SplitKey,
    SplitKeys,
    FoldIn,
    Dropout,
    UniformLike,
}

impl KeyPrimitive {
    pub const ALL: [Self; 5] = [
        Self::SplitKey,
        Self::SplitKeys,
        Self::FoldIn,
        Self::Dropout,
        Self::UniformLike,
    ];

    /// The zero-based source operand every key primitive takes its key at.
    pub const KEY_OPERAND: usize = 0;

    /// The builtin that spells this primitive in source.
    pub const fn builtin(self) -> &'static str {
        match self {
            Self::SplitKey => "split_key",
            Self::SplitKeys => "split_keys",
            Self::FoldIn => "fold_in",
            Self::Dropout => "dropout",
            Self::UniformLike => "uniform_like",
        }
    }

    pub fn of_builtin(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|primitive| primitive.builtin() == name)
    }
}

/// The allow-list: every operation a key-carrying value may reach.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KeyAdmission {
    /// A key primitive consumes the key at its key operand.
    Primitive(KeyPrimitive),
    /// [05-OP-67]: `drop` consumes any value.
    Drop,
    /// A runtime branch's join: the arms of an `if` or a `match`, and
    /// `KeySelect` in the graph.
    Join,
    /// Construction and destructuring of tuples, records and data values.
    Aggregate,
    /// A binding, or a block's, a handler region's or a function's result:
    /// the value moves unchanged to its one new owner.
    Move,
    /// A builtin case that routes each value of a type parameter to one
    /// consumer (`OneConsumer`).
    RoutingCase,
    /// A call through a parameter whose declared type carries a key: a user
    /// function, a closure or a constructor, called directly or through
    /// `grad`, `vmap` or `jit`.
    KeyParameter,
    /// A graph root: the key a graph returns to its caller.
    Root,
    /// A read of a key tensor's extent: `shape` or `numel` in source, and in
    /// the graph any input slot an operation reads only for its extent
    /// (`chelis_dag_core::verify::slot_read`). The extent is not key material, so
    /// the read is not a use and leaves the key live.
    ExtentObservation,
}

impl KeyAdmission {
    pub const ALL: [Self; 13] = [
        Self::Primitive(KeyPrimitive::SplitKey),
        Self::Primitive(KeyPrimitive::SplitKeys),
        Self::Primitive(KeyPrimitive::FoldIn),
        Self::Primitive(KeyPrimitive::Dropout),
        Self::Primitive(KeyPrimitive::UniformLike),
        Self::Drop,
        Self::Join,
        Self::Aggregate,
        Self::Move,
        Self::RoutingCase,
        Self::KeyParameter,
        Self::Root,
        Self::ExtentObservation,
    ];

    /// Whether the admitted operation stays an operation of the lowered
    /// graph, where the verifier's key rules read it. Construction,
    /// destructuring and moves flatten into the graph's values, calls
    /// inline, and routing cases run in the host lane, so none of those is
    /// a graph operation.
    pub const fn in_graph(self) -> bool {
        match self {
            Self::Primitive(_) | Self::Drop | Self::Join | Self::Root | Self::ExtentObservation => {
                true
            }
            Self::Aggregate | Self::Move | Self::RoutingCase | Self::KeyParameter => false,
        }
    }
}
