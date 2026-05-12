module Std.Init.Xavier
export (sample)
sig sample: tensor[32, 128, p] -> p -> tensor[32, 128, p] ! { Random }
