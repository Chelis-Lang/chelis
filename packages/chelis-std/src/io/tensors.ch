module Std.Io.Tensors
export (TensorArchive, open_hnw, read_f64, read_f32, read_f16, read_bf16, read_i64, read_i32, read_i16, read_i8, read_bool)
import Std.Io.Json (Json, parse_json, json_get, json_string, json_int, json_array)
import Std.Text (join)
-- One stored tensor: its name, its dtype spelled as a Chelis primitive, its
-- row-major shape, the absolute byte offset and byte length of its
-- little-endian payload in the archive file, and the lowercase hexadecimal
-- SHA-256 of that payload.
type TensorEntry =
  | TensorEntry { name: string, dtype: string, shape: List[i64], offset: i64, byte_len: i64, sha256: string }
-- An open archive: its path for diagnostics, its mapping, and its entries.
@opaque
type TensorArchive =
  | TensorArchive { path: string, mapped: MappedFile, entries: List[TensorEntry] }
def archive_fail[t](path: string, detail: string) -> t = fail(join(["Std.Io.Tensors: ", path, ": ", detail], ""))
def require(path: string, condition: bool, detail: string) -> unit = if condition then () else archive_fail(path, detail)
def required[a](path: string, field: string, value: Option[a]) -> a =
  match value with {
    | Some(found) => found
    | None => archive_fail(path, string_concat("manifest field is missing or malformed: ", field))
  }
def u32_le(m: MappedFile, offset: i64) -> i64 = {
  b = mmap_read(m, offset, 4i64)
  add(add(index(b, 0i64), mul(index(b, 1i64), 256i64)), add(mul(index(b, 2i64), 65536i64), mul(index(b, 3i64), 16777216i64)))
}
def nibble_text(value: i64) -> string = string_slice("0123456789abcdef", value, 1i64)
def digest_text(bytes: List[i64]) -> string = join(map(fn (byte: i64) -> string_concat(nibble_text(floor_div(byte, 16i64)), nibble_text(mod(byte, 16i64))), bytes), "")
def align_up(value: i64, alignment: i64) -> i64 = mul(floor_div(add(value, sub(alignment, 1i64)), alignment), alignment)
def json_i64s(path: string, field: string, items: List[Json]) -> List[i64] = map(fn (item: Json) -> required(path, field, json_int(Some(item))), items)
def hnw_entry(path: string, payload_start: i64, item: Json) -> TensorEntry = {
  name = required(path, "tensors[].id", json_string(json_get(item, "id")))
  layout = required(path, "tensors[].layout", json_string(json_get(item, "layout")))
  encoding = required(path, "tensors[].encoding", json_string(json_get(item, "encoding")))
  _ = require(path, eq(layout, "onnx-row-major"), join(["tensor ", name, " has unsupported layout ", layout], ""))
  _ = require(path, eq(encoding, "raw-le"), join(["tensor ", name, " has unsupported encoding ", encoding], ""))
  shape = json_i64s(path, "tensors[].shape", required(path, "tensors[].shape", json_array(json_get(item, "shape"))))
  offset = required(path, "tensors[].offset", json_int(json_get(item, "offset")))
  byte_len = required(path, "tensors[].byte_len", json_int(json_get(item, "byte_len")))
  _ = require(path, eq(len(filter(fn (d: i64) -> lt(d, 0i64), shape)), 0i64), join(["tensor ", name, " has a negative extent"], ""))
  _ = require(path, gte(offset, 0i64), join(["tensor ", name, " has a negative offset"], ""))
  _ = require(path, gte(byte_len, 0i64), join(["tensor ", name, " has a negative byte length"], ""))
  TensorEntry { name, dtype: required(path, "tensors[].dtype", json_string(json_get(item, "dtype"))), shape, offset: add(payload_start, offset), byte_len, sha256: required(path, "tensors[].sha256", json_string(json_get(item, "sha256"))) }
}
-- Open a hydronnx `.hnw` weight archive (format major 1). The manifest is
-- checked against its header checksum before it is read; each tensor's own
-- checksum is checked when the tensor is read.
def open_hnw(path: string) -> TensorArchive ! { IO } = {
  m = mmap_file(path)
  _ = require(path, gte(mmap_len(m), 80i64), "truncated fixed header")
  _ = require(path, eq(mmap_read(m, 0i64, 7i64), [72i64, 78i64, 88i64, 87i64, 71i64, 84i64, 0i64]), "bad magic")
  _ = require(path, eq(index(mmap_read(m, 7i64, 1i64), 0i64), 1i64), "unsupported major version")
  header_len = u32_le(m, 8i64)
  manifest_len = u32_le(m, 12i64)
  _ = require(path, gte(header_len, 80i64), "header length is shorter than the fixed header")
  _ = require(path, eq(mmap_sha256(m, header_len, manifest_len), digest_text(mmap_read(m, 16i64, 32i64))), "manifest checksum mismatch")
  manifest = parse_json(mmap_text(m, header_len, manifest_len))
  format = required(path, "format", json_get(manifest, "format"))
  _ = require(path, eq(required(path, "format.name", json_string(json_get(format, "name"))), "hydronnx-weights"), "not a hydronnx-weights archive")
  _ = require(path, eq(required(path, "format.major", json_int(json_get(format, "major"))), 1i64), "unsupported major version")
  payload_start = align_up(add(header_len, manifest_len), 64i64)
  items = required(path, "tensors", json_array(json_get(manifest, "tensors")))
  TensorArchive { path, mapped: m, entries: map(fn (item: Json) -> hnw_entry(path, payload_start, item), items) }
}
def find_entry(archive: TensorArchive, name: string) -> TensorEntry = {
  found = filter(fn (entry: TensorEntry) -> eq(entry.name, name), archive.entries)
  if eq(len(found), 0i64) then archive_fail(archive.path, string_concat("no tensor named ", name)) else index(found, 0i64)
}
-- The checks every read makes before it touches the payload: the stored
-- dtype is the one read, the stored shape is the declared one, the byte
-- length is the shape's, and the payload's checksum is the recorded one.
-- Returns the payload's element count.
def checked_count(archive: TensorArchive, entry: TensorEntry, dtype: string, width: i64, dims: List[i64]) -> i64 = {
  path = archive.path
  _ = require(path, eq(entry.dtype, dtype), join(["tensor ", entry.name, " has dtype ", entry.dtype, ", read as ", dtype], ""))
  _ = require(path, eq(entry.shape, dims), join(["tensor ", entry.name, " has shape [", join(map(fn (d: i64) -> to_string(d), entry.shape), ", "), "], read as [", join(map(fn (d: i64) -> to_string(d), dims), ", "), "]"], ""))
  count = fold(fn (acc: i64, d: i64) -> mul(acc, d), 1i64, dims)
  _ = require(path, eq(entry.byte_len, mul(count, width)), join(["tensor ", entry.name, " has byte length ", to_string(entry.byte_len), ", its shape needs ", to_string(mul(count, width))], ""))
  _ = require(path, eq(mmap_sha256(archive.mapped, entry.offset, entry.byte_len), entry.sha256), join(["tensor ", entry.name, " checksum mismatch"], ""))
  count
}
-- Each reader returns the named tensor's elements in row-major order as a
-- rank-one tensor. `dims` is the shape the caller declares; a reader fails
-- unless the archive stores exactly that shape at exactly that dtype with
-- an intact checksum. `reshape(read_f32(a, name, dims), dims)` under a
-- declared tensor type gives the tensor its declared shape.
def read_f64(archive: TensorArchive, name: string, dims: List[i64]) -> tensor[*, f64] = {
  entry = find_entry(archive, name)
  mmap_tensor(archive.mapped, entry.offset, checked_count(archive, entry, "f64", 8i64, dims), f64)
}
def read_f32(archive: TensorArchive, name: string, dims: List[i64]) -> tensor[*, f32] = {
  entry = find_entry(archive, name)
  mmap_tensor(archive.mapped, entry.offset, checked_count(archive, entry, "f32", 4i64, dims), f32)
}
def read_f16(archive: TensorArchive, name: string, dims: List[i64]) -> tensor[*, f16] = {
  entry = find_entry(archive, name)
  mmap_tensor(archive.mapped, entry.offset, checked_count(archive, entry, "f16", 2i64, dims), f16)
}
def read_bf16(archive: TensorArchive, name: string, dims: List[i64]) -> tensor[*, bf16] = {
  entry = find_entry(archive, name)
  mmap_tensor(archive.mapped, entry.offset, checked_count(archive, entry, "bf16", 2i64, dims), bf16)
}
def read_i64(archive: TensorArchive, name: string, dims: List[i64]) -> tensor[*, i64] = {
  entry = find_entry(archive, name)
  mmap_tensor(archive.mapped, entry.offset, checked_count(archive, entry, "i64", 8i64, dims), i64)
}
def read_i32(archive: TensorArchive, name: string, dims: List[i64]) -> tensor[*, i32] = {
  entry = find_entry(archive, name)
  mmap_tensor(archive.mapped, entry.offset, checked_count(archive, entry, "i32", 4i64, dims), i32)
}
def read_i16(archive: TensorArchive, name: string, dims: List[i64]) -> tensor[*, i16] = {
  entry = find_entry(archive, name)
  mmap_tensor(archive.mapped, entry.offset, checked_count(archive, entry, "i16", 2i64, dims), i16)
}
def read_i8(archive: TensorArchive, name: string, dims: List[i64]) -> tensor[*, i8] = {
  entry = find_entry(archive, name)
  mmap_tensor(archive.mapped, entry.offset, checked_count(archive, entry, "i8", 1i64, dims), i8)
}
def read_bool(archive: TensorArchive, name: string, dims: List[i64]) -> tensor[*, bool] = {
  entry = find_entry(archive, name)
  mmap_tensor(archive.mapped, entry.offset, checked_count(archive, entry, "bool", 1i64, dims), bool)
}
