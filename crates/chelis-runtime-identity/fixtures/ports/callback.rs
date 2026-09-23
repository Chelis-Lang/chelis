use chelis_runtime_identity::*;
pub fn rejected(recipe: &RuntimeRecipe) {
    let callback = || std::fs::read("input").unwrap();
    let _ = derive_descriptor(recipe, &callback);
}
