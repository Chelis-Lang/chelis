use chelis_unord::UnordMap;

fn main() {
    let mut map = UnordMap::<String, usize>::new();
    let _ = map.all(|_, _| true);
    let _ = map.any(|_, _| false);
    let _ = map.count(|_, _| true);
    map.for_each_mut(|_, _| {});
    map.retain(|_, _| true);
    let _ = map.clone().map_values(|value| value);
    let _ = map.to_sorted_by_key(|key, _| key.clone());
    let _ = map.into_sorted_by_key(|key, _| key.clone());
}
