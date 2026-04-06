# Implementation of a deep learning library in Futhark

## Metadata
- **Authors:** Duc Minh Tran
- **Venue/Year:** BSc Thesis, University of Copenhagen, 2018

## Summary
This thesis explores the implementation of a deep learning library in Futhark, a data-parallel functional language designed for generating efficient GPU code. The work investigates whether Futhark is expressive enough to handle the complex nature of deep learning libraries, given its limitations on language semantics. The thesis presents a novel approach to representing neural networks as compositions of functions, which avoids the limitations of non-regular arrays and demonstrates that Futhark's module and type system can provide the necessary abstraction. The library is benchmarked against TensorFlow, showing competitive performance for larger batch sizes on multilayer perceptrons but falling short on smaller batch sizes and convolutional networks. The thesis concludes that Futhark is capable of implementing a deep learning library with high flexibility and maintainability, and that future improvements in the Futhark compiler and better convolutional algorithms could further enhance performance.

## Key Contributions
- **Novel Network Representation:** Introduces a representation of neural networks as compositions of functions, avoiding the limitations of non-regular arrays in Futhark.
- **Expressive Module System:** Demonstrates that Futhark's module and type system can provide the necessary abstraction for deep learning libraries.
- **Benchmarking Results:** Provides performance comparisons between the Futhark library and TensorFlow, highlighting strengths and areas for improvement.
- **Implementation Details:** Offers detailed insights into the implementation of various components, including activation functions, loss functions, optimizers, and layers.

## Technical Approach
The thesis presents a comprehensive approach to implementing a deep learning library in Futhark. It starts by discussing the limitations of Futhark, such as the inability to handle irregular arrays and arrays of functions. To overcome these limitations, the thesis proposes representing neural networks as compositions of functions, where each layer is defined by two functions: one for the forward pass and one for the backward pass. This approach avoids the need for auxiliary information and allows for a more natural representation of neural networks. The implementation includes various components such as activation functions, loss functions, optimizers, and layers (dense, convolutional, max-pooling, and flatten). The thesis also discusses the challenges of implementing convolutional layers efficiently in Futhark and proposes using the GEMM approach with the im2col function.

## Results
The thesis benchmarks the Futhark library against TensorFlow on two types of networks: multilayer perceptrons (MLPs) and convolutional networks. The results show that for MLPs with larger batch sizes, the Futhark library performs comparably to TensorFlow. However, for smaller batch sizes and convolutional networks, the Futhark library falls behind. The thesis attributes this performance gap to the choice of algorithms and suggests that implementing better convolutional algorithms could improve performance. Overall, the benchmarks demonstrate that Futhark is capable of implementing a deep learning library with competitive performance, especially for larger batch sizes.

## Relevance
This paper is relevant to language design, compilers, and ML systems work because it explores the implementation of a complex machine learning library in a functional, data-parallel language. It demonstrates how Futhark's unique features, such as its module system and type system, can be leveraged to provide the necessary abstraction for deep learning. The thesis also highlights the challenges of implementing efficient convolutional layers in Futhark and suggests potential solutions. The benchmarking results provide insights into the performance of Futhark compared to established deep learning frameworks like TensorFlow, which is valuable for researchers and practitioners in the field.
