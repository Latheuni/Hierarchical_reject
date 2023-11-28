# Uncertainty-aware single-cell annotation with a hierarchical reject option [![Documentation Status](https://readthedocs.org/projects/hierarchical-reject/badge/?version=latest)](https://hierarchical-reject.readthedocs.io/en/latest/?badge=latest) [![GitHub license](https://img.shields.io/github/license/Latheuni/Hierarchical_reject)](https://github.com/Latheuni/Hierarchical_reject/blob/main/LICENSE)

## Description

Code accompanying the paper "Uncertainty-aware single-cell annotation with a hierarchical reject option".

## Content of the repository 

This repository contains all the code necessary to recreate the analyses performed in the paper, given the datasets that can be freely downloaded from their corresponding papers.

This repository contains code for the three main parts of the analyses (that can be found in the corresponding folders)
1. *Preprocessing* contains functions that enable the preprocessing of the datasets so that they are compliable with flat and or hierarchical annotation
2. *Run* contains the code needed to run flat and or hierarchical classification with F-test or HVG feature selection
3. *Evaluation* contains code that allows the evaluation of partial and or full rejection with the help of accuracy-rejection curves

The rejection process itself is implemented inside the hierarchical classification algorithm (the hclf folder) and implemented in the evaluation functions for flat classification. The evaluation functions should output enough information so that further evaluation of the rejection processes besides the accuracy-rejection curves is also possible.

## Installation
This code is designed to be cloned and run locally. A yaml file to create a conda environment is provided (Hierarchical_reject.yml). A local installation of Python and R is needed (the latter to access the loom format of the Flyatlas datasets).

## Documentation
Documentation can be found here: [![Documentation Status](https://readthedocs.org/projects/hierarchical-reject/badge/?version=latest)](https://hierarchical-reject.readthedocs.io/en/latest/?badge=latest)
, together with a tutorial on how to run the analyses.

## Datasets
This paper uses 5 open-source datasets:
1. The Allen Mouse Brain (AMB) dataset [1]
2. The COVID dataset [2]
3. The Azimuth PBMC dataset [3]
4. and 5. from the Flyatlas [4] the Flyhead and Flybody dataset
The data files used in this paper for the analyses are available at Zenodo: (all credits of these datasets go to the original creators of the dataset)
## References
[1] Tasic, B. et al. (2018). Shared and distinct transcriptomic cell types across neocortical areas. Nature, 563 (7729), 72–78. https://doi.org/10.1038/s41586-018-0654-5
[2] Chan Zuckerberg Initiative Single-Cell COVID-19 Consortia et al. (2020). Single cell profiling ofCOVID-19 patients: an international data resource from multiple tissues. Medrxiv preprint. https://doi.org/10.1101/2020.11.20.20227355
[3] Stuart, T. et al. (2019). Comprehensive Integration of Single-Cell Data. Cell, 177(7), 1888–1902.e21. https://doi.org/10.1016/j.cell.2019.05.031
[4] Li, H. et al. (2022). Fly Cell Atlas: A single-nucleus transcriptomic atlas of the adult fruit fly. Science, 375(6584), eabk2432. https://doi.org/10.1126/science.abk2432
