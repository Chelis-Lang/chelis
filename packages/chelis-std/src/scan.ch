module Std.Scan
export (scan_list)
def scan_list[acc, item](step: acc -> item -> acc, initial: acc, values: List[item]) -> List[acc] = scan(step, initial, values)
