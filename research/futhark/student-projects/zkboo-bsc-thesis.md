# ZKBoo on the GPU: Better soundness errors for a little extra

## Metadata
- **Authors:** Nina Andrup Pedersen, Jakob Schneider Villumsen
- **Venue/Year:** Bachelor Report (15 ECTS) in Computer Science, Aarhus University, July 2023

## Summary
This paper presents ZKG, a parallel implementation of ZKBoo's SHA-256 (2,3)-decomposition on the GPU using the Futhark Programming Language. ZKBoo builds on the MPC-in-the-head paradigm to produce Zero-Knowledge proofs from MPC protocols. While ZKG is generally slower than ZKBoo on the CPU in reaching soundness error 2^-80, it achieves soundness error 2^-584 significantly faster on the GPU. Experimental results show that ZKG is up to 40 times faster in achieving soundness error 2^-584, making it more secure against quantum computers.

## Key Contributions
- Implementation of ZKBoo's SHA-256 (2,3)-decomposition on the GPU using Futhark
- Demonstration of improved soundness error (2^-584) compared to ZKBoo (2^-80)
- Up to 40 times faster achievement of soundness error 2^-584 on the GPU
- Analysis of the impact of sequential nature of SHA-256 on performance

## Technical Approach
The paper implements ZKBoo's (2,3)-function decomposition for SHA-256 on the GPU using Futhark. The (2,3)-decomposition allows any function to be transformed into an MPC protocol, which can then be used for a Zero-Knowledge proof. The implementation uses Futhark's parallel programming capabilities to distribute protocol repetitions across parallel tasks on the GPU. The paper also introduces a parallel mock version of SHA-256 to demonstrate the potential performance gains from using data-parallel algorithms.

## Results
The experimental results show that while ZKG is slower than ZKBoo on the CPU in reaching soundness error 2^-80, it achieves soundness error 2^-584 significantly faster on the GPU. Specifically, ZKG is up to 40 times faster in achieving soundness error 2^-584. The paper also demonstrates that the sequential nature of SHA-256 significantly impacts performance, and using a parallel mock version of SHA-256 greatly improves performance.

## Relevance
This paper is relevant to language design, compilers, and ML systems work because it demonstrates the potential of using GPU parallelization to improve the performance of Zero-Knowledge proof protocols. The use of Futhark, a functional programming language designed for parallel programming, highlights the importance of language design in enabling efficient parallel computation. The paper also provides insights into the challenges of implementing cryptographic protocols on parallel hardware and the potential benefits of using data-parallel algorithms.
