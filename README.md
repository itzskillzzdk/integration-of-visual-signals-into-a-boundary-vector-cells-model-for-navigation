# Visual-BVC: Integration of Optic Flow for Bio-inspired Navigation

![Python](https://img.shields.io/badge/python-3.8.10-blue) ![CARLA](https://img.shields.io/badge/CARLA%20Simulator-0.9.14-orange)

## About

This project proposes an innovative approach to autonomous navigation inspired by biological mechanisms observed in mammals. It focuses specifically on the hippocampal system, which uses several types of specialized cells for spatial coding, such as place cells, grid cells, head direction cells, and Boundary Vector Cells (BVCs).

BVCs are particularly interesting because they activate based on the subject's distance and orientation relative to the boundaries of its environment.

To adapt this biological mechanism to artificial navigation, this project explores the use of visual cues. Building on previous work, we use optic flow to provide the necessary information for estimating BVC inputs.

## Objectives

The project is structured around several specific objectives:

* **Research & State of the Art**: Conducting a comprehensive literature review on BVCs, with a focus on those using optic flow and visual models in general.
* **Realistic Simulation**: Setting up the CARLA simulation environment to generate high-fidelity visual data (and potentially from other sensors).
* **Processing Pipeline**: Implementing the complete computational chain: `Optic Flow` $\rightarrow$ `Environment Boundary Estimation` $\rightarrow$ `BVC Activation`.
* **Evaluation & Multimodality**: Evaluating the visual model's performance and exploring a system combining LIDAR and Camera data.

## Prerequisites

* Python 3.8.10 (recommended)
* [CARLA Simulator](https://carla.org/) (Version 0.9.14)

## References

This project relies on the following scientific literature:

1. O'Keefe, J., & Burgess, N. (1996). Geometric determinants of the place fields of hippocampal neurons. *Nature*, 381(6581), 425-428. https://doi.org/10.1038/381425a0
2. Raudies, F., & Hasselmo, M. E. (2012). Modeling Boundary Vector Cell Firing Given Optic Flow as a Cue. *PLOS Computational Biology*, 8(6), e1002553. https://doi.org/10.1371/journal.pcbi.1002553
3. Lever, C., Burton, S., Jeewajee, A., O'Keefe, J., & Burgess, N. (2009). Boundary Vector Cells in the Subiculum of the Hippocampal Formation. *Journal of Neuroscience*
