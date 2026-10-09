`chelis build --target c` builds a call that passes a runtime-extent tensor to a
parameter whose axes are concrete names, including a generic `def` that returns
a free-dimension ADT. It previously failed in ownership lowering with an
internal call-argument type mismatch, although `chelis check` and `chelis eval`
accepted the program. See [#3388](https://github.com/Chelis-Lang/chelis/issues/3388).
