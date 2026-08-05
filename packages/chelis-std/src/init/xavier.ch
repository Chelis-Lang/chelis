module Std.Init.Xavier
export (sample)
sig sample: tensor[32, 128, p] -> p -> tensor[32, 128, p] ! { Random }
def sample[p](template: tensor[32, 128, p], scale: p) -> tensor[32, 128, p] ! { Random } = fail("Std.Init.Xavier.sample is not implemented")
