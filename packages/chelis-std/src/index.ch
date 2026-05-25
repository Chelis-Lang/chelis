module Std.Index
export (list_index, take_list, drop_list)
def list_index[item](values: List[item], idx: int64) -> item = index(values, idx)
def take_list[item](values: List[item], count: int64) -> List[item] = take(values, count)
def drop_list[item](values: List[item], count: int64) -> List[item] = drop(values, count)
