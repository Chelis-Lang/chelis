# FShark: Futhark Programming in FSharp

## Metadata
- **Authors:** Mikkel Storgaard Knudsen
- **Venue/Year:** Master's Thesis, University of Copenhagen, 2018

## Summary
This thesis presents FShark, a high-level programming language that allows developers to write GPU-accelerated computational kernels using a subset of F#. FShark programs are automatically translated to Futhark, a purely functional language designed for efficient GPU code generation. The thesis also introduces a C# code generator for Futhark, enabling seamless integration of Futhark-generated GPU kernels into C# and F# projects. The work demonstrates the feasibility of this approach through comprehensive testing and benchmarking, showing that FShark-generated GPU kernels perform comparably to handwritten Futhark code.

## Key Contributions
- A C# code generator for the Futhark compiler, enabling the generation of GPU-accelerated libraries for C# and F#.
- A subset of the F# language that can be automatically translated to Futhark, along with a standard library implementing Futhark's data-parallel operators in F#.
- A compiler and wrapper pipeline for compiling F# modules to GPU-accelerated libraries and integrating them into F# projects.
- A comprehensive test suite and benchmarks demonstrating the correctness and performance of FShark-generated GPU kernels.

## Technical Approach
The thesis describes the architecture and implementation of the FShark language and compiler. The FShark compiler translates F# code into Futhark source code, which is then compiled to C# using the Futhark C# code generator. The generated C# code can be used as a library in C# and F# projects. The thesis also discusses the design choices and challenges in implementing the FShark language, including array handling, memory management, and interoperability with Futhark.

## Results
The thesis presents extensive testing and benchmarking results to evaluate the correctness and performance of the FShark language and compiler. The results show that FShark-generated GPU kernels perform comparably to handwritten Futhark code, with performance within 0-3% for complex benchmarks. The thesis also demonstrates that FShark allows developers to write complex GPU benchmarks in an idiomatic F# style.

## Relevance
This paper is relevant to language design, compilers, and ML systems work because it presents a novel approach to integrating GPU programming into mainstream languages. The FShark language and compiler provide a high-level interface for writing GPU kernels, making GPU programming more accessible to developers. The C# code generator for Futhark enables seamless integration of Futhark-generated GPU kernels into C# and F# projects, expanding the reach of Futhark to a wider audience. The work also contributes to the field of compiler design by presenting a new approach to translating between high-level and low-level languages for GPU programming.
