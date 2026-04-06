# Massively Parallel Selection of Stable History Period in Change Detection for Time Series Data with Missing Values

## Metadata
- **Authors:** Nikolaj Hey Hinnerskov
- **Venue/Year:** BSc Thesis, University of Copenhagen, 2021

## Summary
This thesis presents a massively parallel implementation of stable history detection for change detection in satellite time series data with missing values. The work focuses on implementing the reversely ordered CUSUM (ROC) test and recursive residuals algorithms, which are essential components of the BFAST-Monitor method for detecting environmental disturbances like deforestation. The implementation is written in Futhark, a data-parallel array language, and targets GPUs for acceleration. The thesis also addresses numerical stability issues in linear model fitting by implementing a rank-revealing QR decomposition. The implementation is validated against the R reference implementation and achieves significant speedups compared to existing code.

## Key Contributions
- Algorithmic descriptions of the ROC test and recursive residuals for stable history detection
- Data-parallel implementation in Futhark targeting GPUs
- Integration with the BFAST-Monitor implementation by Gieseke et al.
- Implementation of a linear model fitting library in Futhark using rank-revealing QR decomposition
- Translation of LINPACK QR-decomposition routines from FORTRAN to Futhark
- Validation against the R reference implementation

## Technical Approach
The thesis implements the ROC test and recursive residuals algorithms for stable history detection. The ROC test evaluates cumulative sums of prediction errors against a boundary function to detect structural breaks in time series data. Recursive residuals are standardized one-step-ahead prediction errors used in the ROC test. The implementation addresses numerical stability issues in linear model fitting by using rank-revealing QR decomposition instead of direct matrix inversion. The code is written in Futhark and exploits both outer parallelism (across pixels) and inner parallelism (within pixel computations) through padding techniques to handle missing values.

## Results
The implementation achieves significant speedups compared to the R reference implementation, with runtimes up to 13 times faster for the complete BFAST-Monitor pipeline. The map-distributed version of the code is 1.3-2.5 times faster than the inner-sequential version. The recursive residuals computation dominates the runtime, accounting for over 80% of the total time. The implementation utilizes about half of the available hardware bandwidth, indicating room for optimization.

## Relevance
This work is highly relevant to language design, compilers, and ML systems work because it demonstrates how to implement complex statistical algorithms in a data-parallel language like Futhark. The thesis addresses challenges in handling missing values, numerical stability, and efficient parallelization, which are crucial considerations for developing high-performance ML systems. The implementation of rank-revealing QR decomposition and the translation of LINPACK routines provide valuable insights into numerical linear algebra for parallel computing.
