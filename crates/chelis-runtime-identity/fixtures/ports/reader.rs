use chelis_runtime_identity::*;
pub fn rejected(recipe: &RuntimeRecipe, reader: &mut dyn std::io::Read) {
    let _ = derive_descriptor(recipe, reader);
}
