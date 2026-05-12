module Std.Nn.Conv
export (conv1d, conv2d_small)
sig conv1d: &tensor[1, 4, 1, 16, p] -> &tensor[8, 4, 1, 3, p] -> tensor[1, 8, 1, 14, p]
sig conv2d_small: &tensor[1, 3, 8, 8, p] -> &tensor[8, 3, 3, 3, p] -> tensor[1, 8, 6, 6, p]
