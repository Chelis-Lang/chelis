listing = print(string_concat("names:", fold(fn (acc: string, name: string) -> string_concat(acc, string_concat("/", name)), "", list_dir("."))))
