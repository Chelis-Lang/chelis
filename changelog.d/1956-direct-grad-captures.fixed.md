Host gradients whose original operand directly names a checked function
declaration, without a captured binding for that target name, now read free
tensor values from the declaration environment instead of caller-local shadows.
Required captures reuse successful initialization and preserve entered errors;
optional shape queries do not initialize absent declarations. Argument order
and anonymous creation-time captures are unchanged. This does not extend alias
or captured-target admission, general gradient support, or backend coverage.
