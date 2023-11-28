**Functions_Accureacy_Reject.py** contains the functions needed to make obtain the results to make accuracy-reject curves.
 - ```Evaluate_AR```: for hierarchical annotation
 - ```Evaluate_AR_Flat```: for flat annotation
 - ```Evaluate_AR_parallel```: for hierarchical annotation for one model that is parallelized

The 'parallel' version of the function exist as for some classifiers the analysis can take quite some time as they are not optimally parallelised which can lead to long runtimes especially on large datasets. Therefore, I made this specific versions of the function. It is not always necessary to use them.
