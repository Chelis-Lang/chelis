module Demo.Main
import Std.Io.Csv (read_csv)
import Std.Io.Json (load_json, json_bool, json_float, json_get, json_int)
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
text_view = print(text)
row_count_view = print(len(rows))
enabled_view = print(enabled)
lr_view = print(lr)
max_length_view = print(max_length)
pad_value_view = print(pad_value)
