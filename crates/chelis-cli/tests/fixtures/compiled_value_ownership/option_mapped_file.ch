def mapped_length(path: string) -> int64 ! { IO } = {
  value: Option[MappedFile] = Some(mmap_file(path))
  match value with {
    | None => 0i64
    | Some(mapped) => mmap_len(mapped)
  }
}
out = mapped_length("__MAPPED_FILE_PATH__")
