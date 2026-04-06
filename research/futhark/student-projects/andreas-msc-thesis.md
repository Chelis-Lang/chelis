# Using Automatic Differentiation to find gradients for recurrent neural networks in Futhark

## Metadata
- **Authors:** Andreas Nicolaisen
- **Venue/Year:** Master Thesis, December 20, 2021

## Summary
This thesis explores the application of backwards automatic differentiation (AD) to compute gradients for recurrent neural networks (RNNs) implemented in the Futhark programming language. The work focuses on transforming an Elman RNN implementation into a program capable of calculating gradients for training purposes. The author develops transformation rules for Futhark constructs, implements both forward and backward traces, and validates the results against PyTorch. The study demonstrates that backwards AD can be effectively applied to RNNs in Futhark using checkpointing strategies to manage memory usage while maintaining computational efficiency.

## Key Contributions
- Development of backwards AD transformation rules for Futhark language constructs including loops, map, and reduce operations
- Implementation of Elman RNN with both forward prediction and gradient computation capabilities
- Validation of gradient calculations against PyTorch implementation
- Performance evaluation of the Futhark implementation on GPU hardware
- Discussion of checkpointing strategies for managing memory in RNN backpropagation

## Technical Approach
The thesis employs backwards automatic differentiation to compute gradients for RNNs. The approach involves creating transformation rules for Futhark constructs:

1. **Loop Transformation**: For do-loops, the forward trace checkpoints loop-variant values at each iteration, while the backward trace recomputes intermediate values using these checkpoints
2. **Map Transformation**: For maps without free variables, the backward trace simply propagates adjoints through the mapped function
3. **Matrix Operations**: For matrix-vector and matrix-matrix multiplications, the author derives explicit adjoint computation rules based on the mathematical properties of these operations

The Elman RNN implementation consists of nested loops: an outer loop iterating over time steps and an inner loop iterating over network layers. The backward trace reverses this order, computing gradients layer by layer from output to input.

## Results
The implementation successfully computes gradients that match those produced by PyTorch for the same network configurations. Performance tests on a RTX 2080 Ti GPU show that the Futhark implementation scales well with input dimensionality, approaching linear growth despite the quadratic complexity inherent to Elman RNNs. This performance is attributed to better GPU utilization with larger parallel workloads.

## Relevance
This work is highly relevant to language design, compilers, and ML systems for several reasons:

1. **Language Extension**: It demonstrates how to extend a functional array language with automatic differentiation capabilities, providing insights for language designers
2. **Compiler Transformations**: The transformation rules developed serve as examples of how compilers can automatically generate gradient-computing code from existing programs
3. **High-Performance ML**: The implementation shows how Futhark's parallel computing capabilities can be leveraged for efficient neural network training
4. **Memory-Computation Tradeoffs**: The checkpointing strategy illustrates important considerations in balancing memory usage against computational redundancy in automatic differentiation

The thesis provides a practical example of implementing advanced ML functionality in a functional array language, bridging the gap between theoretical AD concepts and practical implementation challenges.
