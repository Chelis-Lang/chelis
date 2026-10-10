Programs can load typed tensors from files at run time. `mmap_tensor(mapped,
offset, count, T)` reads `count` little-endian elements of the dtype `T` from a
mapped file as a rank-one tensor, reinterpreting the bits without conversion, for
every data element dtype; `mmap_text` decodes a mapped byte range as UTF-8, and
`mmap_sha256` returns its SHA-256 as lowercase hexadecimal. All three are pure
reads of an open mapping and run in `chelis eval` and compiled C.
`Std.Io.Tensors` opens a hydronnx `.hnw` weight archive with `open_hnw` and reads
each tensor with `read_f32`, `read_f64`, and the other per-dtype readers, which
fail before reading an element unless the archive stores exactly the declared
dtype and shape with an intact checksum. See
[#359](https://github.com/Chelis-Lang/chelis/issues/359).
