# Auto-tuning of threshold-parameters in Futhark

## Metadata
- **Authors:** Frederik Thorøe
- **Venue/Year:** University of Copenhagen, June 2018

## Summary
This report describes the development and implementation of a domain-specific auto-tuner for the Futhark programming language. Futhark is a purely functional language that supports nested data-parallelism and in-place array updates, designed to make GPU programming more accessible by shifting the burden of generating efficient GPU code from programmers to a heavily optimizing compiler. The auto-tuner focuses on optimizing threshold-parameters that determine which version of parallelized code should be executed at runtime. The report presents a new auto-tuner that significantly outperforms the existing general-purpose tuner by leveraging domain-specific knowledge about Futhark's threshold comparisons and execution paths. The new tuner is evaluated on three benchmark programs and demonstrates speedups ranging from 1.9 to 25.5 compared to the previous tuner.

## Key Contributions
- Analysis of threshold comparisons and their dependencies in Futhark programs
- Development of a comparison-based tuner that exhaustively searches through all possible configurations
- Introduction of a branch-based tuner that eliminates redundant configurations using execution path information
- Creation of a final combined tuner that integrates domain-specific knowledge with hill-climbing techniques
- Implementation of compiler modifications to expose threshold comparison information
- Evaluation showing significant improvements in tuning speed and quality of results

## Technical Approach
The technical approach involves several key innovations:

1. **Threshold Comparison Analysis**: The tuner extracts information about all threshold comparisons made during program execution, including the values being compared and the dependencies between comparisons.

2. **Configuration Space Reduction**: Instead of searching through a large space of arbitrary threshold values, the tuner generates configurations based on actual comparison values observed across all datasets. This reduces the search space from potentially millions of values to a manageable number of meaningful configurations.

3. **Execution Path Filtering**: By tracking which execution paths are taken for different configurations, the tuner eliminates redundant configurations that would result in the same code being executed.

4. **Timeout Optimization**: The tuner calculates appropriate timeout values for each program to avoid wasting time on configurations that cause timeouts.

5. **Hybrid Approach**: The final tuner combines exhaustive search for simple programs with hill-climbing techniques for complex programs, using domain-specific knowledge to guide the search.

## Results
The evaluation demonstrates significant improvements over the previous auto-tuner:

- **Speedup**: The new tuner achieves speedups of 1.9x to 25.5x compared to the old tuner
- **Configuration Quality**: The tuned configurations perform at least as well as those found by the old tuner, with a 50% improvement for the most complex program
- **Completeness**: The tuner successfully completes tuning for all three benchmark programs within 30 minutes, whereas the old tuner failed to complete for any of them
- **Search Space Reduction**: The number of configurations tried is reduced by factors ranging from 27.2 to 171.9 depending on the program complexity

## Relevance
This work is highly relevant to language design, compilers, and ML systems for several reasons:

1. **Domain-Specific Optimization**: It demonstrates how incorporating domain-specific knowledge can dramatically improve auto-tuning performance, which is applicable to other domains beyond Futhark.

2. **Compiler-Integrated Auto-tuning**: The approach shows how auto-tuning can be integrated into the compilation process through compiler modifications that expose relevant information.

3. **Performance Portability**: By automatically tuning threshold parameters, the approach helps achieve performance portability across different hardware architectures without requiring manual intervention.

4. **Hybrid Static-Dynamic Analysis**: The work contributes to the broader trend of combining static and dynamic analysis in compilers, showing how runtime information can guide optimization decisions.

5. **Practical Impact**: The significant speedups and improved configuration quality demonstrate the practical value of sophisticated auto-tuning approaches for real-world applications.
