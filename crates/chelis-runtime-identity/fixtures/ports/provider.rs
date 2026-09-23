use chelis_runtime_identity::*;
pub trait Provider { fn capture(&self) -> Vec<CapturedInput>; }
pub fn rejected(recipe: &RuntimeRecipe, provider: &dyn Provider) {
    let _ = derive_descriptor(recipe, provider);
}
