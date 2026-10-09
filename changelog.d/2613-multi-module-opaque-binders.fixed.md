`chelis prove` can fuzz a package property with invariant-bearing opaque
binders from different modules. Generated sample values are constructed in
each type's defining module, while authored code remains subject to ordinary
opaque access checks. See [#2613](https://github.com/Chelis-Lang/chelis/issues/2613).
