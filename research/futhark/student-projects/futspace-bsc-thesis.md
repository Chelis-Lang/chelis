# FutSpace: A Parallelizable Implementation of the Voxel Space Rendering Algorithm

## Metadata
- **Authors:** Jonas Kristensen, Mathias Rasmussen, Jens Sørensen, Christian Troelsen
- **Venue/Year:** Bachelor Project, University of Copenhagen, 2020

## Summary
This project examines the Voxel Space rendering algorithm, a 2.5D graphics technique from the 1990s, and explores how data parallelism can be applied to improve its performance. The authors implement a parallel version of the algorithm in Futhark, a functional parallel programming language, and analyze its runtime characteristics. They also introduce several improvements to the original algorithm, including shadow rendering, bilinear filtering, and the ability to render arbitrary mathematical functions.

## Key Contributions
- A parallel implementation of the Voxel Space rendering algorithm in Futhark
- A detailed analysis of the algorithm's runtime characteristics using work and span
- Several improvements to the original algorithm, including shadow rendering and bilinear filtering
- An interactive application for exploring the rendered environments

## Technical Approach
The authors start by explaining the Voxel Space algorithm and its relationship to ray casting. They then describe how the algorithm can be parallelized using Futhark's second-order array combinators (SOACs). The implementation involves generating color-height pairs for each point in a depth-width-shaped discrete space and then rendering a frame on screen by processing each column of color-height pairs in parallel. The authors use scan and scatter operations to handle occlusion and fill in missing colors.

## Results
The authors benchmark the sequential and parallel performance of their implementation and find that it scales well with increasing screen width and render distance. They also show that the parallel implementation achieves significant speedup compared to the sequential version.

## Relevance
This paper is relevant to language design, compilers, and ML systems work because it demonstrates how a classic graphics algorithm can be parallelized using a functional programming language. The authors' analysis of the algorithm's runtime characteristics using work and span is also valuable for understanding the performance of parallel programs. Additionally, the improvements they introduce to the original algorithm, such as shadow rendering and bilinear filtering, are relevant to graphics and rendering research.
