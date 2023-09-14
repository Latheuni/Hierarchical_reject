```diff
- text in red
+ text in green
! text in orange
# text in gray
@@ text in purple (and bold)@@
```



General_analysis.py contains the functions necessary to run the analysis in the paper:
- Functions for feature selection
   - -Fselection-: for F-test based feature selection
   - HVGselection: for highly variable gene feature selection performed with scanpy
- Functions for running hierarchical and flat annotation
  - without K-fold cross-validation
    - Run_H_NoKF: hierarchical annotation (feature selection is not included)
    - Run_H_NoKF_sparse: hierarchical annotation with input data in a sparse data format (feature selection is not included)
    - Run_Flat_NoKF: flat annotation (feature selection is not included)
  - with K-fold cross-validation
    - for flat annotation: 
      - *Run_Flat_KF*
      - Run_Flat_KF_sparse, 
