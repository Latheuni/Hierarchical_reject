<div align="center">
<h1>Hierarchical reject</h1>


Code accompanying the paper "Uncertainty-aware single-cell annotation with a hierarchical reject option".

![Documentation](https://readthedocs.org/projects/hierarchical_reject/badge/?version=latest&style=flat-default)(https://hierarchical_reject.readthedocs.io/en/latest/index.html)

</div>

This repository contains all the code necessary to recreate the analyses performed in the paper, given the datasets that can be freely downloaded from their corresponding papers.

This repository contains code for the three main parts of the analyses (that can be found in the corresponding folders)
1. *Preprocessing* contains functions that enable the preprocessing of the datasets so that they are compliable with flat and or hierarchical annotation
2. *Run* contains the code needed to run flat and or hierarchical classification with F-test or HVG feature selection
3. *Evaluation* contains code that allows the evaluation of partial and or full rejection with the help of accuracy-rejection curves

The rejection process itself is implemented inside the hierarchical classification algorithm (the hclf folder) and implemented in the Evaluation functions for Flat classification. The Evaluation functions shoudl output enough information so that further evaluation outside of the accuracy-rejection functions is also possible.

There is also documentation of all the usefull functions present at 'https://hierarchical_reject.readthedocs.io/en/latest/index.html', together with a tutorial for more information.