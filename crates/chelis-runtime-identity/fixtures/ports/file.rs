use chelis_runtime_identity::*;
pub fn rejected(recipe: &RuntimeRecipe, file: &mut std::fs::File) {
    let _ = derive_descriptor(recipe, file);
}
