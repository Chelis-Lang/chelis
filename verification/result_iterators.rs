#![feature(try_trait_v2)]

// The map's mutable capture and the borrowed iterator must both survive collect,
// including the item that produced Err. The resumed poll is counted as well.
pub fn collect_resume(len: usize, error_at: usize) -> (bool, usize, Option<usize>, usize) {
    let mut calls = 0;
    let (success, value, next) = {
        let mut iter = (0..len).map(|index| {
            calls += 1;
            if index == error_at {
                Err(100 + index)
            } else {
                Ok(index + 1)
            }
        });
        let collected: Result<Vec<usize>, usize> = iter.by_ref().collect();
        let (success, value) = match collected {
            Ok(values) => (true, values.into_iter().fold(0, |sum, item| sum + item)),
            Err(error) => (false, error),
        };
        let next = match iter.next() {
            Some(Ok(value)) => Some(value),
            Some(Err(error)) => Some(error),
            None => None,
        };
        (success, value, next)
    };
    (success, value, next, calls)
}

pub struct First(pub Option<usize>);

impl FromIterator<usize> for First {
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        Self(iter.into_iter().next())
    }
}

pub fn early_consumer() -> (bool, usize, Option<usize>, usize) {
    let mut calls = 0;
    let (success, value, next) = {
        let mut iter = (0..4).map(|index| {
            calls += 1;
            if index == 2 {
                Err(72)
            } else {
                Ok(index + 1)
            }
        });
        let collected: Result<First, usize> = iter.by_ref().collect();
        let (success, value) = match collected {
            Ok(First(Some(value))) => (true, value),
            Ok(First(None)) => (true, 0),
            Err(error) => (false, error),
        };
        let next = match iter.next() {
            Some(Ok(value)) => Some(value),
            Some(Err(error)) => Some(error),
            None => None,
        };
        (success, value, next)
    };
    (success, value, next, calls)
}

pub struct Reopen;

impl FromIterator<usize> for Reopen {
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        let mut iter = iter.into_iter();
        // The lower bound is erased even before an error is encountered.
        if iter.size_hint() != (0, Some(5)) {
            panic!("initial shunt hint");
        }
        if iter.next() != None {
            panic!("first error must end this poll");
        }
        if iter.size_hint() != (0, Some(0)) {
            panic!("residual hint");
        }
        if iter.next() != Some(2) {
            panic!("shunt must not fuse after an error");
        }
        if iter.size_hint() != (0, Some(0)) {
            panic!("success must not clear residual");
        }
        match iter.next() {
            None | Some(3) => (),
            _ => panic!("third poll"),
        }
        if iter.next() != Some(4) {
            panic!("shunt must resume after another error");
        }
        if iter.size_hint() != (0, Some(0)) {
            panic!("residual must persist");
        }
        Self
    }
}

pub fn repeat_after_none(overwrite: bool) -> (usize, Option<usize>, usize) {
    let mut calls = 0;
    let (error, next) = {
        let mut iter = (0..5).map(|index| {
            calls += 1;
            if index == 0 {
                Err(11)
            } else if overwrite && index == 2 {
                Err(33)
            } else {
                Ok(index + 1)
            }
        });
        let collected: Result<Reopen, usize> = iter.by_ref().collect();
        let error = match collected {
            Ok(_) => panic!("successful polls must not clear the residual"),
            Err(error) => error,
        };
        let next = match iter.next() {
            Some(Ok(value)) => Some(value),
            _ => panic!("collection must leave the fifth item untouched"),
        };
        (error, next)
    };
    (error, next, calls)
}

pub struct Signature(pub usize);

impl FromIterator<usize> for Signature {
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        Self(
            iter.into_iter()
                .fold(0, |signature, item| signature * 10 + item),
        )
    }
}

pub struct Dispatch<'a> {
    pub cursor: &'a mut usize,
    pub folds: &'a mut usize,
    pub error_at: usize,
}

impl Iterator for Dispatch<'_> {
    type Item = Result<usize, usize>;

    fn next(&mut self) -> Option<Self::Item> {
        panic!("shunt must select the underlying try_fold override")
    }

    fn try_fold<B, F, R>(&mut self, mut acc: B, mut fold: F) -> R
    where
        F: FnMut(B, Self::Item) -> R,
        R: std::ops::Try<Output = B>,
    {
        *self.folds += 1;
        while *self.cursor < 3 {
            let index = *self.cursor;
            *self.cursor += 1;
            let item = if index == self.error_at {
                Err(40 + index)
            } else {
                Ok(index + 1)
            };
            acc = fold(acc, item)?;
        }
        R::from_output(acc)
    }
}

// Deliberately not an Iterator: Result::from_iter must restore the original
// source's borrows through this IntoIterator conversion, not return Dispatch.
pub struct Source<'a> {
    pub cursor: &'a mut usize,
    pub folds: &'a mut usize,
    pub error_at: usize,
}

impl<'a> IntoIterator for Source<'a> {
    type Item = Result<usize, usize>;
    type IntoIter = Dispatch<'a>;

    fn into_iter(self) -> Self::IntoIter {
        Dispatch {
            cursor: self.cursor,
            folds: self.folds,
            error_at: self.error_at,
        }
    }
}

pub fn next_dispatch() -> (bool, usize, usize, usize) {
    let mut cursor = 0;
    let mut folds = 0;
    let collected = Result::<First, usize>::from_iter(Source {
        cursor: &mut cursor,
        folds: &mut folds,
        error_at: 1,
    });
    let (success, value) = match collected {
        Ok(First(Some(value))) => (true, value),
        Ok(First(None)) => (true, 0),
        Err(error) => (false, error),
    };
    (success, value, cursor, folds)
}

pub fn fold_dispatch(error: bool) -> (bool, usize, usize, usize) {
    let mut cursor = 0;
    let mut folds = 0;
    let collected = Result::<Signature, usize>::from_iter(Source {
        cursor: &mut cursor,
        folds: &mut folds,
        error_at: if error { 1 } else { 9 },
    });
    let (success, value) = match collected {
        Ok(Signature(value)) => (true, value),
        Err(error) => (false, error),
    };
    (success, value, cursor, folds)
}

struct OffsetIterator {
    next_value: usize,
}

impl Iterator for OffsetIterator {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        let value = self.next_value;
        self.next_value += 1;
        Some(value)
    }

    fn by_ref(&mut self) -> &mut Self {
        self.next_value += 10;
        self
    }
}

pub fn overridden_by_ref() -> (Option<usize>, usize) {
    let mut iter = OffsetIterator { next_value: 0 };
    (iter.by_ref().next(), iter.next_value)
}

pub fn result_method_names(value: usize, success: bool) -> (Option<usize>, usize) {
    let result = if success { Ok(value) } else { Err(0usize) };
    (result.ok(), result.unwrap_or(9))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collection_and_resumption() {
        assert_eq!(collect_resume(0, 9), (true, 0, None, 0));
        assert_eq!(collect_resume(4, 9), (true, 10, None, 4));
        assert_eq!(collect_resume(4, 0), (false, 100, Some(2), 2));
        assert_eq!(collect_resume(4, 2), (false, 102, Some(4), 4));
        assert_eq!(early_consumer(), (true, 1, Some(2), 2));
    }

    #[test]
    fn residual_survives_further_polls() {
        assert_eq!(repeat_after_none(false), (11, Some(5), 5));
        assert_eq!(repeat_after_none(true), (33, Some(5), 5));
    }

    #[test]
    fn specialized_try_fold_and_source_restoration() {
        assert_eq!(next_dispatch(), (true, 1, 1, 1));
        assert_eq!(fold_dispatch(false), (true, 123, 3, 1));
        assert_eq!(fold_dispatch(true), (false, 41, 2, 1));
    }

    #[test]
    fn reference_override_retains_its_effects() {
        assert_eq!(overridden_by_ref(), (Some(10), 11));
    }

    #[test]
    fn result_method_namespace() {
        assert_eq!(result_method_names(5, true), (Some(5), 5));
        assert_eq!(result_method_names(5, false), (None, 9));
    }
}
