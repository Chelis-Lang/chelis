use chelis_unord::{UnordMap, UnordSet};

fn main() {
    let map = UnordMap::<String, usize>::new();
    let set = UnordSet::<String>::new();
    let _ = map.iter();
    let _ = map.keys();
    let _ = map.values();
    let _ = map.clone().into_iter();
    let _ = set.iter();
    let _ = set.clone().into_iter();
    for _ in &map {}
}
