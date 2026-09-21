module Std.Index
export (list_index, take_list, skip_list)
def list_index[item](values: List[item], idx: i64) -> item = index(values, idx)
def take_list[item](values: List[item], count: i64) -> List[item] = take(values, count)
def skip_list[item](values: List[item], count: i64) -> List[item] = skip(values, count)
