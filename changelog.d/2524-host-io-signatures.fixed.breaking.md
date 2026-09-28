The file, mapped-file and process builtins carry the exact signatures
[05-OP-60] and [05-OP-38] give them: `read_file`, `write_file`, `read_lines`,
`read_bytes`, `file_exists`, `list_dir` and `mmap_file` take `string` paths,
`mmap_read` and `mmap_len` take a `MappedFile` and exact `i64` offsets and
counts, and `process_run` takes a `string` and a `List[string]`. Previously any
argument type checked with score 1, including an `f32` path or one typed by a
`Float` binder, and `process_run("echo", "hi")` failed only in `eval`. See
[#2524](https://github.com/Chelis-Lang/chelis/issues/2524).
