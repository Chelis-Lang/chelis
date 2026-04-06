# FFT Generator in Futhark

## Metadata
- **Authors:** Kasper Abildtrup Hansen
- **Venue/Year:** Project in Computer Science, University of Copenhagen, 2018

## Summary
This project investigates the application of FFTW (Fastest Fourier Transform in the West) techniques to implement a new FFT library in Futhark, a data-parallel functional array language. The author develops a prototype library called genfft that incorporates multiple FFT algorithms (radix-2, 3, 4, and Rader's FFT for prime numbers) and compares its performance against Futhark's existing radix-2 FFT library. The results show that genfft outperforms the existing library on large datasets with composite sizes involving radices 2, 3, and 4, but performs poorly on datasets with prime factors due to high memory usage and planning overhead.

## Key Contributions
- Implementation of a prototype FFT library (genfft) in Futhark using FFTW techniques
- Development of multiple FFT algorithms including radix-2, 3, 4, and Rader's FFT
- Creation of a planner component that decomposes input sizes and creates execution plans
- Comprehensive benchmark testing comparing genfft against Futhark's existing FFT library
- Validation of correctness through linearity tests and expected output comparisons

## Technical Approach
The implementation follows the FFTW approach of combining multiple efficient FFT algorithms into a composite algorithm. The system consists of three main components: (1) FFT algorithms implemented as parametric modules, (2) a planner that analyzes input size N and creates a plan based on factor decomposition and a predefined strategy, and (3) an executor that combines the planner and FFTs to compute the DFT. The planner uses a strategy that prioritizes radix-4 for large datasets, then radix-2 and 3, and finally Rader's FFT for prime factors. The implementation leverages Futhark's parallel capabilities while handling the inherently sequential nature of composite FFT computation through for-loops.

## Results
Benchmark tests reveal that genfft outperforms Futhark's existing FFT library on large datasets (N > 3.98 × 10^6) when the input sizes involve composite factors of radices 2, 3, and 4. However, genfft performs significantly worse on datasets with prime factors, sometimes failing to compute the DFT due to excessive memory usage. The planning overhead is substantial, with planning times ranging from 229µs to 3135µs for small datasets, though this overhead diminishes as dataset size increases. The results confirm that while the new FFT implementations are fast, they struggle to compete with the simpler existing library on smaller datasets.

## Relevance
This work is relevant to language design and compiler development for parallel functional languages, demonstrating both the potential and challenges of implementing sophisticated numerical algorithms in Futhark. The project highlights the trade-offs between algorithmic complexity and practical performance, particularly regarding planning overhead and memory usage. The findings suggest that while FFTW techniques can be successfully adapted to Futhark, careful consideration must be given to the balance between planning sophistication and execution efficiency, especially for smaller datasets. The work also points to future improvements such as implementing additional radix FFTs, optimizing prime number handling, and potentially incorporating wisdom-like features for storing optimal plans.
