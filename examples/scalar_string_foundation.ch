epoch_text = " 7 "
loss_text = "0.125"
epoch = match to_int(epoch_text) with {
  | Some(n) => n
  | None => 0i64
}
loss = match to_float(loss_text) with {
  | Some(value) => value
  | None => 1.0f64
}
checkpoint = string_trim("  ckpt  ")
path = string_concat(checkpoint, string_concat("-", epoch |> to_string |> string_concat(".safetensors")))
stem = string_slice(path, 5, 1)
under_threshold = (loss < 0.5f64)
has_ckpt = string_contains(path, "ckpt")
has_prefix = string_starts_with(path, "ckpt-")
has_suffix = string_ends_with(path, ".safetensors")
stem_ok = (stem == "7")
should_stop = and(under_threshold, and(has_ckpt, and(has_prefix, and(has_suffix, stem_ok))))
progress = "progress: " |> string_concat(path) |> print
status = if should_stop then "stop" else "keep-going"
