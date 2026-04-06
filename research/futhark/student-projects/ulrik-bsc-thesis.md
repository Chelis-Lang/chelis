# Optimizing the kNN algorithm for GPGPUs in Futhark using kd-trees, a novel boundary-test, and more!

## Metadata
- **Authors:** Ulrik Elmelund Petersen
- **Venue/Year:** Bachelor thesis, University of Copenhagen, 2020

## Summary
This thesis presents a data-parallel implementation of the k-nearest neighbors (kNN) algorithm in the Futhark language, targeting efficient execution on GPUs. The implementation uses a kd-tree data structure to partition the reference points and reduce the search space, achieving significant speedups compared to the brute-force approach. The thesis introduces several optimizations, including parallel kd-tree construction with padding, a novel boundary-test for pruning the search space, and sorting queries based on their last-visited leaves to improve temporal locality. The implementation is evaluated on various datasets with different dimensions and numbers of queries, demonstrating substantial speedups, especially for lower-dimensional data.

## Key Contributions
- Parallelized kd-tree construction using padding to regularize inner parallelism.
- Novel boundary-test for more accurate pruning of the search space.
- Sorting queries based on their last-visited leaves to improve temporal locality.
- Two-stage iterative traversal of the kd-tree to avoid recursion.
- Comprehensive evaluation of the implementation on various datasets.

## Technical Approach
The thesis starts with a sequential Python implementation of the kNN algorithm using a kd-tree and rewrites it in Futhark using parallel constructs like map, reduce, and scan. The kd-tree is constructed in parallel using a padding technique to ensure regular inner parallelism, allowing the Futhark compiler to generate efficient GPU code. The traversal of the kd-tree is split into two stages: an initialization step to find the natural leaf for each query and an iterative loop to traverse the tree and find the nearest neighbors. The boundary-test is used to decide whether to visit a sibling node, reducing the number of false-positive visits. Queries are sorted based on their last-visited leaves to improve temporal locality.

## Results
The implementation is evaluated on datasets with varying dimensions (d = 5, 7, 9, 12, 16) and numbers of queries (up to 2 million). The results show significant speedups compared to the brute-force approach, with the best speedup of 488x achieved for 2 million queries and reference points with d = 5 and k = 5. The speedups decrease as the dimensionality increases, but the implementation still outperforms the brute-force approach for higher dimensions. The boundary-test is shown to reduce the number of visited leaves, especially for higher dimensions, improving the algorithm's scalability.

## Relevance
This paper is relevant to language design, compilers, and ML systems work because it demonstrates how to efficiently implement a complex algorithm like kNN on GPUs using a high-level language like Futhark. The optimizations introduced, such as parallel kd-tree construction and the boundary-test, can be applied to other algorithms that use spatial search structures. The thesis also highlights the importance of considering the curse of dimensionality when designing algorithms for high-dimensional data.
