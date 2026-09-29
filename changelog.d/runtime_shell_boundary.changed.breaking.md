The bundled standard library exposes no `Std.Tokenizer` or `Std.Init.*`
modules. School provides normal, Kaiming, Xavier, and truncated-normal
initializers under `School.Init.*`; the illustrative CSV and JSON pipeline
uses only runtime data APIs. This boundary change is part of
[chelis#2303](https://github.com/Chelis-Lang/chelis/issues/2303).
