`mmap_read` traps `Domain` when the requested range ends past the mapping, in
`chelis eval` and in compiled C. It previously returned only the bytes that
existed. `Std.Io.read_head_bytes` still returns the first
`min(count, file_length)` bytes. See
[#359](https://github.com/Chelis-Lang/chelis/issues/359).
