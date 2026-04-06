# Not-Quite-Supersonic AD: Implementing forward and reverse mode automatic differentiation in the Futhark interpreter

## Metadata
- **Authors:** Marcus Jensen
- **Venue/Year:** Bachelor's thesis, August 12th, 2024

## Summary
This thesis presents the implementation of forward and reverse mode automatic differentiation (AD) in the Futhark interpreter. The author incrementally improves the interpreter's AD capabilities until it matches the Futhark compiler's functionality. The implementation is tested for correctness against the existing test suite and benchmarked for performance. The thesis concludes with recommendations for future work needed before merging the implementation into the main Futhark codebase.

## Key Contributions
- Complete implementation of forward mode AD in the Futhark interpreter
- Complete implementation of reverse mode AD in the Futhark interpreter
- Support for nested AD calls to compute higher-order derivatives
- Integration of AD with Futhark's existing value representation system
- Comprehensive testing and benchmarking of the implementation

## Technical Approach
The implementation follows the standard approach to AD by augmenting scalar values with derivative information. For forward mode, values are wrapped in JvpValue structures containing primal values and derivatives. For reverse mode, values are wrapped in VjpValue structures containing computation graphs (tapes). The implementation handles nested AD calls by introducing depth tracking to distinguish between different levels of differentiation.

The core technical challenges addressed include:
- Modifying the interpreter's value representation to support AD
- Implementing doOp to handle primitive operations on AD values
- Creating evalJvp and evalVjp functions to initiate forward and reverse mode AD
- Supporting type conversions for AD values
- Handling composite values (arrays, records, tuples) in AD operations

## Results
The implementation passes 136 out of 138 relevant tests in the Futhark AD test suite. The two failing tests involve non-derivable operations (comparisons and iota), where the correct behavior is undefined. Benchmarking shows:
- No performance degradation when running programs without AD
- Forward mode AD introduces approximately 2.6x slowdown
- Reverse mode AD introduces approximately 2.1x slowdown

## Relevance
This work is highly relevant to language design and compiler development for machine learning systems. Futhark is specifically designed for high-performance numerical computing and machine learning workloads, making efficient AD implementation crucial. The techniques demonstrated here—particularly the handling of nested AD calls and the integration with an existing interpreter—provide valuable insights for other language implementers working on AD support. The performance characteristics and implementation challenges documented in this thesis will be useful for anyone implementing AD in functional programming languages or interpreters.
