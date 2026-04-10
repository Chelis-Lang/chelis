let epoch_text = " 7 "
let loss_text = "0.125"
let epoch = match to_int(epoch_text) with {
  | Some(n) => n
  | None => cast(0, int64)
}
let loss = match to_float(loss_text) with {
  | Some(value) => value
  | None => cast(1.0, f64)
}
let checkpoint = string_trim("  ckpt  ")
let path = string_concat(checkpoint, string_concat("-", string_concat(to_string(epoch), ".safetensors")))
let stem = string_slice(path, 5, 1)
let under_threshold = (loss < cast(0.5, f64))
let has_ckpt = string_contains(path, "ckpt")
let has_prefix = string_starts_with(path, "ckpt-")
let has_suffix = string_ends_with(path, ".safetensors")
let stem_ok = (stem == "7")
let should_stop = and(under_threshold, and(has_ckpt, and(has_prefix, and(has_suffix, stem_ok))))
let progress = print(string_concat("progress: ", path))
let status = if should_stop then "stop" else "keep-going"
