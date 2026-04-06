# Parallel implementations of machine learning algorithms: Gradient boosted decision trees

## Metadata
- **Authors:** Kristian Høi
- **Venue/Year:** University of Copenhagen, March 1, 2021

## Summary
This thesis explores the implementation of Gradient Boosted Decision Trees (GBDT) using parallel computing techniques, specifically targeting GPU acceleration. The author presents a detailed analysis of the XGBoost algorithm and translates it into a parallel framework using Futhark, a high-level functional programming language designed for GPU programming. The implementation is compared against the popular XGBoost framework, demonstrating similar accuracy but with a performance gap that narrows as the number of histogram bins increases. The thesis also discusses the challenges of irregular parallelism and how flattening techniques are used to maximize GPU utilization.

## Key Contributions
- Derivation of the mathematical theory behind gradient boosting and the XGBoost algorithm.
- Translation of the GBDT algorithm into pseudo-Futhark using parallel building blocks.
- Implementation of the algorithm in Futhark, leveraging flattening to remove irregular parallelism.
- Improvement of the algorithm using histograms to reduce computation time.
- Empirical evaluation comparing the Futhark implementation with XGBoost, showing similar accuracy but a performance gap that narrows with more histogram bins.

## Technical Approach
The thesis describes the process of translating the sequential GBDT algorithm into a parallel framework using Futhark's parallel building blocks. The author addresses the challenge of irregular parallelism by using flattening techniques to ensure maximum GPU utilization. The implementation includes the use of histograms to optimize the search for the best split within decision trees, reducing the need for expensive sorting operations. The thesis also discusses the use of segmented operations and the challenges of implementing them efficiently in Futhark.

## Results
The empirical evaluation demonstrates that the Futhark implementation achieves similar accuracy to XGBoost on the Higgs dataset. However, the training time is approximately twice as long as XGBoost for 5 million training instances. The performance gap narrows as the number of histogram bins increases, indicating that the Futhark implementation scales well with more complex models. The thesis also highlights the importance of optimizing histogram creation, which is a significant bottleneck in the implementation.

## Relevance
This paper is highly relevant to language design, compilers, and ML systems work. It provides insights into the challenges of implementing machine learning algorithms on GPUs using high-level programming languages. The use of Futhark and the techniques for handling irregular parallelism are particularly relevant for researchers and practitioners working on optimizing ML algorithms for parallel execution. The thesis also contributes to the broader discussion on the trade-offs between accuracy and performance in machine learning implementations.
