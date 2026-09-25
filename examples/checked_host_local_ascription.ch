def pad_pair(width: i64) -> tensor[*, *, i64] = {
  ids = (pad_sequences_to([[1i64, 2i64]], width, 0i64) : tensor[1, 2, i64])
  ids
}
def main() -> tensor[*, *, i64] = pad_pair(2i64)
