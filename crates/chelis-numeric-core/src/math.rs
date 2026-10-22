//! IEEE operations absent from the core float interface.

pub(crate) trait FloatMath: Sized {
    fn sqrt(self) -> Self;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn trunc(self) -> Self;
    fn fract(self) -> Self;
    fn round_ties_even(self) -> Self;
    #[cfg(feature = "alloc")]
    fn mul_add(self, multiplier: Self, addend: Self) -> Self;
}

macro_rules! ieee_math {
    ($ty:ty, $sqrt:ident, $floor:ident, $ceil:ident, $trunc:ident, $round:ident, $fma:ident) => {
        impl FloatMath for $ty {
            #[inline]
            fn sqrt(self) -> Self {
                libm::$sqrt(self)
            }
            #[inline]
            fn floor(self) -> Self {
                libm::$floor(self)
            }
            #[inline]
            fn ceil(self) -> Self {
                libm::$ceil(self)
            }
            #[inline]
            fn trunc(self) -> Self {
                libm::$trunc(self)
            }
            #[inline]
            fn fract(self) -> Self {
                self - libm::$trunc(self)
            }
            #[inline]
            fn round_ties_even(self) -> Self {
                libm::$round(self)
            }
            #[cfg(feature = "alloc")]
            #[inline]
            fn mul_add(self, multiplier: Self, addend: Self) -> Self {
                libm::$fma(self, multiplier, addend)
            }
        }
    };
}

ieee_math!(f32, sqrtf, floorf, ceilf, truncf, roundevenf, fmaf);
ieee_math!(f64, sqrt, floor, ceil, trunc, roundeven, fma);
