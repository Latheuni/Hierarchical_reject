<div align="center">
<h1>Hierarchical reject</h1>


Code accompanying the paper "Uncertainty-aware single-cell annotation with a hierarchical reject option".

[![Documentation Status](https://readthedocs.org/projects/hierarchical-reject/badge/?version=latest)](https://hierarchical-reject.readthedocs.io/en/latest/?badge=latest)


</div>

This repository contains all the code necessary to recreate the analyses performed in the paper, given the datasets that can be freely downloaded from their corresponding papers.

This repository contains code for the three main parts of the analyses (that can be found in the corresponding folders)
1. *Preprocessing* contains functions that enable the preprocessing of the datasets so that they are compliable with flat and or hierarchical annotation
2. *Run* contains the code needed to run flat and or hierarchical classification with F-test or HVG feature selection
3. *Evaluation* contains code that allows the evaluation of partial and or full rejection with the help of accuracy-rejection curves

The rejection process itself is implemented inside the hierarchical classification algorithm (the hclf folder) and implemented in the evaluation functions for flat classification. The evaluation functions should output enough information so that further evaluation of the rejection processes besides the accuracy-rejection curves is also possible.

Documentation can be found here: [![Documentation Status](https://readthedocs.org/projects/hierarchical-reject/badge/?version=latest)](https://hierarchical-reject.readthedocs.io/en/latest/?badge=latest)
, together with a tutorial on how to run the analyses.