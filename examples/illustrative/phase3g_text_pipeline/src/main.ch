module Demo.Main
import Std.Io.Csv (read_csv)
import Std.Io.Json (load_json, json_bool, json_float, json_get, json_int)
import Std.Tokenizer (batch_encode, decode, encode, load_tokenizer)
rows: List[Dict[string, string]] = read_csv("train.csv")
first_row = index(rows, cast(0, i64))
text = match dict_get(first_row, "text") with {
  | Some(value) => value
  | None => ""
}
cfg = load_json("config.json")
max_length = match json_int(json_get(cfg, "max_length")) with {
  | Some(n) => n
  | None => cast(8, i64)
}
pad_value = match json_int(json_get(cfg, "pad_value")) with {
  | Some(n) => n
  | None => cast(0, i64)
}
enabled = match json_bool(json_get(cfg, "enabled")) with {
  | Some(flag) => flag
  | None => false
}
lr = match json_float(json_get(cfg, "learning_rate")) with {
  | Some(rate) => rate
  | None => cast(0.0, f64)
}
tok = load_tokenizer("tokenizer.json")
tokens = encode(tok, text)
decoded = decode(tok, tokens)
batch = batch_encode(tok, [text, "hello"], max_length, pad_value)
decoded_view = print(decoded)
shape_view = print((shape(batch, 0), shape(batch, 1)))
enabled_view = print(enabled)
lr_view = print(lr)
token_count_view = print(len(tokens))
