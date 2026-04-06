# Application of Probabilistic Machine Learning Methods for Protein Generation

## Metadata
- **Authors:** Peter Kanstrup Larsen
- **Venue/Year:** MSc Thesis, University of Copenhagen, 2023

## Summary
This thesis explores the application of denoising diffusion probabilistic models (DDPMs) for protein generation, implemented in the Futhark programming language. The work demonstrates how a diffusion model can be implemented in Futhark and compares its performance to an equivalent PyTorch implementation. The research focuses on implementing key components of DDPMs including forward and backward diffusion processes, neural network architectures (LeNet and U-Net), and various layers such as convolutional, pooling, and normalization layers. The thesis validates the implementation through benchmarks, visual inspection, and classification of generated images using a CNN classifier.

## Key Contributions
- Implementation of a diffusion model in Futhark using both LeNet and U-Net architectures
- Development of convolutional neural network layers (convolution, pooling, dense) in Futhark
- Implementation of group normalization and positional embeddings
- Comparative performance analysis between Futhark and PyTorch implementations
- Validation of generated images through visual inspection, SSIM, HOG, and classification
- Exploration of Futhark-AD for automatic differentiation

## Technical Approach
The thesis implements the core components of DDPMs including:
- **Forward Diffusion**: Implements the noise addition process using closed-form sampling
- **Backward Diffusion**: Implements the denoising process using a neural network to predict noise
- **Neural Networks**: Implements both LeNet-5-Fut and U-Net architectures with convolutional, pooling, and dense layers
- **Group Normalization**: Implements group normalization for stabilizing training
- **Positional Embeddings**: Implements sinusoidal positional embeddings for time information
- **Training**: Implements the training loop using the DDPM objective function and Adam optimizer

## Results
The implementation achieves:
- Successful generation of MNIST digits with recognizable quality
- Performance comparisons showing Futhark's efficiency for small input sizes
- Classification accuracy of around 90% for generated digits using a trained CNN
- SSIM scores around 0.27 for generated vs. real images
- HOG descriptor similarity between generated and real images
- Nearest neighbor analysis showing generated images are distinct from training data

## Relevance
This work is significant for language design, compilers, and ML systems because:
- It demonstrates the viability of Futhark for implementing complex ML models
- It provides insights into the performance characteristics of functional array programming languages for deep learning
- It explores the use of automatic differentiation in Futhark through Futhark-AD
- It bridges the gap between high-performance computing and machine learning
- It shows potential applications of functional programming in scientific computing domains like protein design
