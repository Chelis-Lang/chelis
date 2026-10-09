`chelis build --target c` lowers an explicit borrow passed in host position
the way `chelis eval` does: a borrowed local record such as `f(x, &c)`, and
a borrow such as `print(&x)` inside a rank-polymorphic function. It
previously refused them with `unsupported: Deep expression `borrow` on host
expression lowering`. See
[#3383](https://github.com/Chelis-Lang/chelis/issues/3383).
