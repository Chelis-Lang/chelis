# Ray Tracing for Sensor Simulation using Parallel Functional Programming

## Metadata
- **Authors:** Johan Johansson & Ari von Nordenskjöld
- **Venue/Year:** Master's thesis, Chalmers University of Technology, 2020

## Summary
This thesis presents a physically-based ray tracer implemented in the parallel functional programming language Futhark, designed to simulate both vision and LIDAR sensors for automotive applications. The authors demonstrate that Futhark's high-level abstractions and automatic parallelization capabilities can produce efficient GPU code while maintaining code clarity and domain-specific focus. The ray tracer incorporates advanced features like multiple importance sampling, spectral path tracing, and physically-based material models to achieve accurate sensor simulation. The implementation is validated through relative measurements comparing virtual scenes to real-world data, establishing a framework for systematic improvement of the simulation accuracy.

## Key Contributions
- Implementation of a physically-based ray tracer in Futhark that can simulate both camera and LIDAR sensors
- Integration of multiple importance sampling and spectral path tracing for improved accuracy and convergence
- Development of a material hierarchy supporting diffuse, dielectric, and conductor materials with microfacet models
- Creation of a validation framework using relative metrics for systematic improvement
- Demonstration of Futhark's effectiveness for complex parallel algorithms in computer graphics

## Technical Approach
The ray tracer uses path tracing with Monte Carlo integration to solve the rendering equation. Key technical components include:
- Ray casting with bounding volume hierarchies (BVH) for efficient scene traversal
- Importance sampling techniques to reduce variance in Monte Carlo integration
- Multiple importance sampling (MIS) combining BSDF and light source sampling
- Spectral path tracing with wavelength-dependent material properties
- Material hierarchy supporting diffuse, dielectric, and conductor materials with microfacet models
- Direct lighting integration for improved performance with small light sources

## Results
The implementation was benchmarked against four Futhark backends (sequential C, multicore C, OpenCL, and CUDA) on two scenes (Cornell box and sphere box). GPU backends showed significant performance improvements over CPU implementations, with CUDA achieving 21s and 49s for the respective scenes compared to 805s and 1079s for sequential C. Validation was performed by comparing LIDAR point clouds and camera renders against real-world data, establishing relative metrics (RMS distance of 0.074 for LIDAR and SSIM of 0.728033 for camera) for future improvements.

## Relevance
This work is highly relevant to language design, compilers, and ML systems as it demonstrates the practical application of a high-level functional programming language for complex parallel algorithms in computer graphics. The thesis shows how Futhark's abstractions can hide low-level GPU programming details while still producing efficient code, making it accessible to domain experts without requiring deep knowledge of GPU architecture. The validation framework and modular design also provide insights into building reliable simulation systems for autonomous vehicle development.
