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
