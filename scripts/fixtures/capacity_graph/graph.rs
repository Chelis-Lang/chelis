#![allow(dead_code)]
mod hidden {
    pub struct Payload<T> {
        value: T,
    }
    pub type Alias<T> = Payload<T>;
}
pub use hidden::Payload as Published;
pub enum Tree<T> {
    Value(T),
    Branch(Vec<Tree<T>>),
}
pub struct Root {
    values: Option<Box<hidden::Alias<f64>>>,
    tree: Tree<i64>,
    label: String,
}

pub struct Hex<const DIGITS: usize>(String);
pub struct SizedPayload<T, const WIDTH: usize> {
    values: [T; WIDTH],
    marker: Hex<WIDTH>,
}
pub struct ConstRoot {
    small: Hex<4>,
    medium: Hex<8>,
    wide: Hex<16>,
    numeric: SizedPayload<i32, 4>,
}
