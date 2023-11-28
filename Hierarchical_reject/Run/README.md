



**General_analyses.py** contains the functions necessary to run the analyses in the paper:
- Functions for feature selection
   - ```Fselection```: for F-test based feature selection
   - ```HVGselection```: for highly variable gene feature selection performed with scanpy
- Functions for running hierarchical and flat annotation
  - without K-fold cross-validation
    - ```Run_H_NoKF``` hierarchical annotation (feature selection is not included)
    - ```Run_H_NoKF_sparse```: hierarchical annotation with input data in a sparse data format (feature selection is not included)
    - ```Run_Flat_NoKF```: flat annotation (feature selection is not included)
  - with K-fold cross-validation
    - for flat annotation: 
      - ```Run_Flat_KF```
      - ```Run_Flat_KF_sparse```: with sparse input data
      - ```Run_Flat_KF_splitted```: runs only one specified fold within the K-fold cross-validation scheme
      - ```Run_Flat_KF_sparse_splitted```: with sparse input data and only runs one specified fold
    - for hierarchical annotation
      - ```Run_H_KF```
      - ```Run_H_KF_sparse```
- Functions to save the results:
   - ```SaveResultsKF```
 
The 'splitted'versions of the functions exist as for some classifiers the analysis can take quite some time as they are not optimally parallelised which can lead to long runtimes especially on large datasets. It is not always necessary to use them.
