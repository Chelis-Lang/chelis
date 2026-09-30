def mapped_length(path: string) -> i64 ! { IO } = {
  value: Option[MappedFile] = path |> mmap_file |> Some
  match value with {
    | None => 0i64
    | Some(mapped) => mmap_len(mapped)
  }
}
out = mapped_length("__MAPPED_FILE_PATH__")
