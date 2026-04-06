# Reactive Benchmarking

## Metadata
- **Authors:** Aleksander Junge
- **Venue/Year:** Bachelor's Project, University of Copenhagen, 2022

## Summary
This project introduces a new benchmarking technique for the Futhark compiler that automatically determines when to stop benchmarking based on the collected data's reliability. The approach addresses the challenge of non-deterministic program execution times by using a two-phase process: an initial timed phase followed by a convergence phase that monitors autocorrelation and relative standard error (RSE) to decide when sufficient data has been collected. The new tool was evaluated against the previous Futhark benchmarking system and found to provide more reliable results faster, particularly for programs of varying execution times.

## Key Contributions
- Developed a reactive benchmarking tool that automatically determines when to stop benchmarking
- Implemented a two-phase approach: initial timed phase followed by convergence phase
- Used autocorrelation and RSE as metrics to assess data reliability
- Demonstrated improved reliability and speed compared to manual iteration count specification
- Provided confidence intervals using bootstrapping techniques

## Technical Approach
The new benchmarking tool operates in two phases. First, it runs the benchmark for 0.5 seconds (or at least 10 iterations) to establish an initial dataset. Second, it enters a convergence phase where it evaluates the data against specific criteria:

1. **Autocorrelation threshold**: Measures dependencies between consecutive iterations to detect warm-up phases or periodic performance patterns
2. **Relative Standard Error (RSE)**: Normalizes the standard error by the mean to assess measurement precision

The tool discards the initial timed phase if it's deemed too noisy (high autocorrelation) and continues collecting samples until either the RSE falls below a threshold (adjusted based on autocorrelation level) or a maximum time limit is reached. The convergence criteria balance between speed and accuracy by requiring lower RSE when autocorrelation is high.

For statistical analysis, the tool employs bootstrapping to generate confidence intervals, providing users with a measure of the variation in their measurements without requiring parametric distribution assumptions.

## Results
The evaluation compared the new tool against the old Futhark benchmarking system using various iteration counts (10, 10,000, and automatic). Key findings include:

- **Reliability**: The new tool achieved an average RSD of 3.8% across benchmark repetitions, compared to 15% for 10 iterations and 2.4% for 10,000 iterations
- **Speed**: The new tool completed benchmarking in 4 minutes 16 seconds for certain benchmarks, versus 23 minutes 44 seconds for 10,000 iterations
- **Execution time dependency**: Benchmarks under 250µs showed higher variation (6.6% RSD) than those over 250µs (1.2% RSD)
- **Validity**: The new tool's results closely matched those obtained with 10,000 iterations for longer-running programs, with some deviation (up to ~15%) for very short-running programs
- **Autocorrelation reduction**: The new tool reduced average autocorrelation from 0.52 to 0.18 by eliminating warm-up phases

## Relevance
This work is highly relevant to language design, compilers, and ML systems for several reasons:

1. **Automation**: Reduces the burden on developers to manually tune iteration counts for reliable benchmarking
2. **Statistical rigor**: Implements sound statistical methods (bootstrapping, autocorrelation analysis) for performance measurement
3. **Performance characterization**: Provides insights into the reliability and variation of performance measurements
4. **Compiler evaluation**: Enables more accurate comparison of compiler optimizations and code generation strategies
5. **ML system benchmarking**: Addresses the challenges of benchmarking ML workloads with non-deterministic execution patterns

The approach is particularly valuable for evaluating performance changes in compilers and ML systems where small improvements need to be measured reliably against significant noise in execution times.
