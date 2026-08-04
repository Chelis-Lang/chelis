module PipelineRejected.Main

def first(x: tensor[4, f32]) -> tensor[4, f32] ! { } = dropout(x, 0.25)
