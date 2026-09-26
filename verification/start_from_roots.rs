//! Charon's canonical start-from patterns (the roots of `CHARON_CALLGRAPH` and of
//! the whole-core chunks) select the items that Aeneas keeps as roots: inherent
//! and trait impl elements, closures with multi-digit disambiguators, anonymous
//! constants (`_#N`) and comma-joined pattern lists.

pub struct Counter {
    pub value: u32,
}

impl Counter {
    pub fn bump(&mut self) -> u32 {
        self.value += 1;
        self.value
    }
}

pub trait Scale<A> {
    fn scale(&self, by: A) -> u32;
}

impl Scale<u32> for Counter {
    fn scale(&self, by: u32) -> u32 {
        self.value * by
    }
}

impl Scale<u8> for Counter {
    fn scale(&self, by: u8) -> u32 {
        self.value + by as u32
    }
}

/// Eleven closures: the last one is `closure#10`.
pub fn offsets(x: u32) -> u32 {
    let c0 = |v: u32| v;
    let c1 = |v: u32| v + 1;
    let c2 = |v: u32| v + 2;
    let c3 = |v: u32| v + 3;
    let c4 = |v: u32| v + 4;
    let c5 = |v: u32| v + 5;
    let c6 = |v: u32| v + 6;
    let c7 = |v: u32| v + 7;
    let c8 = |v: u32| v + 8;
    let c9 = |v: u32| v + 9;
    let c10 = |v: u32| v * 10;
    c10(c9(c8(c7(c6(c5(c4(c3(c2(c1(c0(x)))))))))))
}

const _: () = ();

const _: () = {
    impl Counter {
        pub fn reset(&mut self) -> u32 {
            let old = self.value;
            self.value = 0;
            old
        }
    }
};

pub fn plain(x: u32) -> u32 {
    x + 1
}

pub fn other(x: u32) -> u32 {
    x * 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_roots() {
        let mut counter = Counter { value: 4 };
        assert_eq!(counter.bump(), 5);
        assert_eq!(Scale::<u32>::scale(&counter, 3), 15);
        assert_eq!(Scale::<u8>::scale(&counter, 3), 8);
        assert_eq!(counter.reset(), 5);
        assert_eq!(counter.value, 0);
        // The chain reaches `closure#10` with 46.
        assert_eq!(offsets(1), 460);
        assert_eq!((plain(3), other(3)), (4, 6));
    }
}
