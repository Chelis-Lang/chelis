module Std.Init.Xavier
export (sample)
sig sample: tensor[32, 128, f32] -> f32 -> tensor[32, 128, f32] ! { Random }
