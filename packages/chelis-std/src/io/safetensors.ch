module Std.Io.Safetensors
export (save_tensors, load_tensors)
sig save_tensors: string -> string -> string
sig load_tensors: string -> string
def save_tensors(path: string, tensors: string) -> string = fail(string_concat("Std.Io.Safetensors.save_tensors is not implemented: ", path))
def load_tensors(path: string) -> string = fail(string_concat("Std.Io.Safetensors.load_tensors is not implemented: ", path))
