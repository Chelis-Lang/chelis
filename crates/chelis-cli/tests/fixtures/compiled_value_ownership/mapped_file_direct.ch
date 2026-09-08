def mapped_length(path: string) -> int64 ! { IO } = {
  mapped = mmap_file(path)
  mmap_len(mapped)
}
out = mapped_length("__MAPPED_FILE_PATH__")
