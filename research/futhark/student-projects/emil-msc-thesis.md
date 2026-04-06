# Support Vector Machines in Futhark

## Metadata
- **Authors:** Emil U. Weihe
- **Venue/Year:** Master's Thesis, University of Copenhagen, 2020

## Summary
This thesis presents a novel GPU-accelerated support vector machine (SVM) library implemented in Futhark, a high-level functional programming language designed for efficient parallel execution on GPUs. The library, called FutharkSVM (FSVM), implements a C-SVM classifier with modular kernel support, allowing users to define custom kernel functions. The implementation uses Sequential Minimal Optimization (SMO) for training and employs a two-level decomposition strategy to handle large datasets that cannot fit entirely in GPU memory. The library was benchmarked against ThunderSVM, a popular GPU-accelerated SVM library, showing competitive performance in training and significantly faster prediction times (2.4-9.6x faster). The results demonstrate that Futhark is a promising language for implementing machine learning algorithms that can compete with established solutions while offering additional benefits like hardware agnosticism and enhanced safety features.

## Key Contributions
- Implementation of a complete SVM library in Futhark with support for multiple kernel functions
- Introduction of modular kernel modules that allow users to define custom kernel functions
- Implementation of a two-level decomposition strategy for handling large datasets
- Development of a GPU-accelerated prediction method using parallel voting
- Comprehensive benchmarking against ThunderSVM showing competitive performance
- Creation of Python bindings for easier integration with data science workflows

## Technical Approach
The library uses Futhark's higher-order module system to create a parametric SVM module that can work with different floating-point precisions and kernel functions. The core training algorithm implements SMO with a two-level decomposition approach: an outer loop selects working sets of data points, and an inner loop performs SMO on these subsets. A least-recently-used (LRU) cache stores kernel matrix rows to improve performance. The prediction function computes outputs for all binary models in parallel and uses segmented reduction to count votes efficiently. The implementation handles multiclass classification using the one-versus-one strategy, where each pair of classes gets its own binary classifier.

## Results
Benchmark results show that FSVM performs comparably to ThunderSVM in training time, with some datasets showing 2.2x improvement while others were slightly slower. For prediction, FSVM consistently outperformed ThunderSVM by 2.4 to 9.6 times. The objective values and biases achieved by both libraries were nearly identical, indicating similar optimization quality. Training errors were slightly higher for FSVM in some cases, potentially due to different tie-breaking strategies. The number of support vectors was identical between both implementations, ensuring fair comparison of prediction times.

## Relevance
This work is highly relevant to language design, compilers, and ML systems as it demonstrates that high-level functional programming languages like Futhark can produce competitive machine learning implementations. The results show that Futhark's abstraction capabilities do not come at the cost of performance, and its hardware-agnostic nature makes it more portable than CUDA-specific solutions. The modular design approach for kernel functions provides a template for extensible ML libraries. Additionally, the success of this implementation suggests that Futhark could be a viable alternative for developing other GPU-accelerated machine learning algorithms, potentially simplifying development while maintaining performance.
