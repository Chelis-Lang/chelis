# Highly parallel algorithms on GPU with Futhark

## Metadata
- **Authors:** Michaël El Kharroubi
- **Venue/Year:** Bachelor thesis, HES-SO, August 2020

## Summary
This bachelor thesis explores the use of Futhark, a functional programming language designed for efficient parallelism, in implementing highly parallel algorithms on GPUs. The author first compares Futhark's performance against traditional CUDA and OpenCL using cellular automata as a benchmark. The results show that Futhark achieves comparable performance without requiring deep hardware-specific optimization. The thesis then focuses on implementing two block ciphers, AES and Camellia, in Futhark. During this process, the author develops a small library called Futhark Host Allocation (FHA) to reduce data transfer times between the host and GPU. Finally, the author creates a file encryption tool using the implemented ciphers and benchmarks it against OpenSSL, demonstrating competitive performance, especially for larger files.

## Key Contributions
- Implementation and performance comparison of cellular automata in Futhark, CUDA, and OpenCL.
- Development of a Futhark library for AES and Camellia block ciphers.
- Creation of the Futhark Host Allocation (FHA) library to optimize data transfers between host and GPU.
- Implementation of a file encryption tool using the developed ciphers and benchmarking against OpenSSL.

## Technical Approach
The thesis begins by introducing the concept of parallelism and its importance in modern computing. It then provides an overview of Futhark, its features, and its approach to parallelization using Second-Order Array Combinators (SOACs). The author implements two cellular automata in Futhark, CUDA, and OpenCL to compare their performance. The results show that Futhark achieves comparable performance without requiring deep hardware-specific optimization.

The thesis then focuses on implementing AES and Camellia block ciphers in Futhark. The author analyzes the algorithms, identifies optimization opportunities, and implements them using Futhark's features. The FHA library is developed to reduce data transfer times between the host and GPU by utilizing pinned memory. Finally, the author creates a file encryption tool using the implemented ciphers and benchmarks it against OpenSSL.

## Results
The performance comparison of cellular automata shows that Futhark achieves comparable performance to CUDA and OpenCL without requiring deep hardware-specific optimization. The implementation of AES and Camellia in Futhark demonstrates the language's capability to handle complex cryptographic algorithms efficiently. The FHA library successfully reduces data transfer times between the host and GPU, improving overall performance. The file encryption tool benchmarks against OpenSSL show competitive performance, especially for larger files.

## Relevance
This thesis is relevant to language design, compilers, and ML systems work in several ways. It demonstrates the potential of functional programming languages like Futhark for efficient parallelism on GPUs. The development of the FHA library showcases the importance of optimizing data transfers in GPU computing. The implementation of block ciphers in Futhark highlights the language's capability to handle complex algorithms efficiently. Overall, this thesis provides valuable insights into the use of Futhark for highly parallel algorithms and its potential for future research and development in the field.
