use serde::{Deserialize, Serialize};

/// A byte-offset range in source text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Span {
    /// Byte offset of the start of this span in the source.
    pub offset: usize,
    /// Byte length of this span.
    pub len: usize,
}

impl Span {
    #[must_use]
    pub fn new(offset: usize, len: usize) -> Self {
        Self { offset, len }
    }

    /// The exclusive end byte offset.
    #[must_use]
    pub fn end(&self) -> usize {
        self.offset + self.len
    }

    /// Create a span covering from `self` through `other`.
    #[must_use]
    pub fn merge(self, other: Span) -> Span {
        let start = self.offset.min(other.offset);
        let end = self.end().max(other.end());
        Span::new(start, end - start)
    }
}
