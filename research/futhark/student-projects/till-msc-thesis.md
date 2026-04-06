# Unsupervised Clustering of Sparse Data in Futhark

## Metadata
- **Authors:** Till Severin Grenzdörffer
- **Venue/Year:** Master's thesis, University of Copenhagen, 2021

## Summary
This thesis investigates the feasibility of using the Futhark programming language to efficiently implement k-means clustering and mixture models on GPUs for large sparse datasets. The author explores different sparse data representations and optimizations for the distance computation and reduction steps of k-means, achieving significant speedups over CPU implementations. The thesis also presents a framework that abstracts from the underlying data representation while maintaining efficiency, enabling the implementation of various mixture models like spherical k-means, Gaussian mixture models, and von Mises-Fisher mixture models.

## Key Contributions
- Exploration of sparse data representations and optimizations for k-means on GPUs
- Implementation of k-means with speedups of at least factor 10 over multicore CPU implementations
- Development of a framework for implementing mixture models on sparse data
- Implementation of Gaussian mixture models with diagonal covariance matrices achieving a speedup of factor 1893 over a single-core CPU implementation

## Technical Approach
The thesis focuses on two main steps in the k-means algorithm: distance computation and reduction. For distance computation, the author explores different sparse data representations like CSR, COO, and ELLPACK, and proposes optimizations like flattening and sequentialization. For the reduction step, the author investigates the use of atomic operations and sorting by cluster assignments. The framework for mixture models abstracts from the data representation and provides operations like semiring operations and reduce operations that can be implemented efficiently for different sparse formats.

## Results
The author benchmarks the k-means implementation on various datasets and achieves significant speedups over CPU implementations. The framework for mixture models is demonstrated to be effective for implementing different models like spherical k-means, Gaussian mixture models, and von Mises-Fisher mixture models. The Gaussian mixture model implementation achieves a speedup of factor 1893 over a single-core CPU implementation.

## Relevance
This thesis is relevant to language design, compilers, and ML systems work as it explores the use of a functional programming language (Futhark) for efficient GPU implementations of machine learning algorithms. The framework for mixture models provides a general approach for implementing these algorithms on sparse data, which is important for many real-world applications. The optimizations and techniques explored in this thesis can be applied to other machine learning algorithms and data structures.
