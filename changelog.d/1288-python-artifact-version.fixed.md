Python's compiled-artifact loader validates the manifest's required ABI version
before decoding metadata, inspecting source files, or loading a shared library.
The writer and loader use the shared compiler API schema so the manifest and its
numeric metadata participate in wire discovery. Missing, duplicate, malformed,
and unsupported versions fail explicitly; valid ABI version 1 metadata retains
its existing JSON representation. Part of
[#1288](https://github.com/Chelis-Lang/chelis/issues/1288).
