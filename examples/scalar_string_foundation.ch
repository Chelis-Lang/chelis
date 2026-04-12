epoch_text = " 7 "
loss_text = "0.125"
epoch = match to_int(epoch_text) with {
  | Some(n) => n
  | None => cast(0, int64)
}
loss = match to_float(loss_text) with {
  | Some(value) => value
  | None => cast(1.0, f64)
}
checkpoint = string_trim("  ckpt  ")
path = string_concat(checkpoint, string_concat("-", string_concat(to_string(epoch), ".safetensors")))
stem = string_slice(path, 5, 1)
under_threshold = (loss < cast(0.5, f64))
has_ckpt = string_contains(path, "ckpt")
has_prefix = string_starts_with(path, "ckpt-")
has_suffix = string_ends_with(path, ".safetensors")
stem_ok = (stem == "7")
should_stop = and(under_threshold, and(has_ckpt, and(has_prefix, and(has_suffix, stem_ok))))
progress = print(string_concat("progress: ", path))
status = if should_stop then "stop" else "keep-going"
