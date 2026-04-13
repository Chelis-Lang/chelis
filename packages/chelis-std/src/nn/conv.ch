module Std.Nn.Conv
export (conv1d, conv2d_small)
def conv1d(x: tensor[1, 4, 1, 16, f32], kernel: tensor[8, 4, 1, 3, f32]) -> tensor[1, 8, 1, 14, f32] = (conv2d(x, kernel, 1, 0) : tensor[1, 8, 1, 14, f32])
def conv2d_small(x: tensor[1, 3, 8, 8, f32], kernel: tensor[8, 3, 3, 3, f32]) -> tensor[1, 8, 6, 6, f32] = (conv2d(x, kernel, 1, 0) : tensor[1, 8, 6, 6, f32])
