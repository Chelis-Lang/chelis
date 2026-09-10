Compiler execution JSON advances to version 3 and WireDag JSON to version 9.
Scalar and tensor carriers preserve exact integers and IEEE floating-point bits,
including signed zero and NaN payloads. Numeric parameters, extents, references
and report fields enforce their owning domains on encoding and decoding.
Readers reject missing, older and future versions; regenerate saved wire
documents with the current compiler. The wire census now requires executed
final authority for every discovered numeric leaf and retains no legacy cohort.
