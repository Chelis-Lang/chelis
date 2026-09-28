-- A host string chooses the numeric branch and carries no cotangent.
def pick[n](w: tensor[n, f32], name: string) -> tensor[n, f32] = if eq(name, "w") then w else neg(w)
def loss[n](w: tensor[n, f32]) -> tensor[f32] = sum(pick(w, "w"), 0i32)
def other_loss[n](w: tensor[n, f32]) -> tensor[f32] = sum(pick(w, "other"), 0i32)
selected = grad(loss)(to_tensor([1.0f32, 2.0f32]))
other = grad(other_loss)(to_tensor([1.0f32, 2.0f32]))
