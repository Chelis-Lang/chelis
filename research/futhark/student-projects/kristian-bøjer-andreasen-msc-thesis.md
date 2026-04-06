# Data-Parallel Coherency Sensitive Hashing for Approximate Nearest Neighbour Fields

## Metadata
- **Authors:** Kristian B. Andreasen
- **Venue/Year:** University of Copenhagen, May 31, 2021

## Summary
This thesis presents a data-parallel implementation of Coherency Sensitive Hashing (CSH) for computing Approximate Nearest Neighbour Fields (ANNFs) using the Futhark programming language. CSH is a method that combines locality sensitivity and coherency to efficiently find similar patches between two images. The data-parallel implementation significantly improves the runtime performance compared to the original CPU-based CSH while maintaining accuracy. The implementation addresses challenges in parallelizing sequential recurrences and candidate ranking, and explores various performance/accuracy trade-offs. Experimental results show that the data-parallel CSH is competitive with other ANNF methods like Propagation-Assisted KD-trees.

## Key Contributions
- Implementation of data-parallel CSH in Futhark
- Solution to parallelizing sequential recurrences using linear function composition and scans
- Solution to parallelizing candidate ranking using intra-group parallelism
- Exploration of performance/accuracy trade-offs including K nearest neighbours, candidate generation, and early stopping
- Experimental evaluation comparing data-parallel CSH with CPU-CSH and Propagation-Assisted KD-trees

## Technical Approach
The thesis implements data-parallel CSH by addressing two critical challenges:

1. **Projecting Walsh-Hadamard Kernels**: The sequential recurrence for computing projections is parallelized using linear function composition and scans. This involves segmenting the recurrence and scanning each segment separately.

2. **Ranking Candidates**: The sequential approach to ranking candidates is parallelized using intra-group parallelism. This involves iterating over two steps: finding the candidate with the lowest distance and comparing it to matches to determine whether to keep it.

The implementation also explores various trade-offs including K nearest neighbours, generating less candidates based on KNN and coherency, and early stopping.

## Results
Experimental results show that the data-parallel CSH implementation achieves significant speedups compared to the original CPU-based CSH while maintaining accuracy. The implementation is competitive with Propagation-Assisted KD-trees in terms of both runtime and accuracy. The results also show that increasing K nearest neighbours improves accuracy but increases runtime, and that reducing the number of candidates generated can improve runtime at the cost of accuracy.

## Relevance
This paper is relevant to language design, compilers, and ML systems work because it demonstrates the effectiveness of data-parallel programming for computer vision tasks. The implementation in Futhark shows how a high-level functional language can be used to generate efficient parallel code for GPUs. The solutions to parallelizing sequential recurrences and candidate ranking are also relevant to other parallel programming tasks.
