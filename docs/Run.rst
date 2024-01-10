Functions to run the analyses:
====================


Feature selection methods
-------------------------
.. automodule:: Run.General_analyses
    :members: Fselection, HVGselection

Functions to run the analyses
-----------------------------

No K-fold cross-validation
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: Run.General_analyses
    :members: Run_H_NoKF, Run_H_NoKF_sparse, Run_Flat_NoKF

With K-fold cross-validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Flat analyses
^^^^^^^^^^^^
.. automodule:: Run.General_analyses
    :members: Run_Flat_KF_sparse, Run_Flat_KF, Run_Flat_KF_sparse_splitted, Run_Flat_KF_splitted

Hierarchical analyses
^^^^^^^^^^^^^^^^^^^^^
.. automodule:: Run.General_analyses
    :members: Run_H_KF, Run_H_KF_sparse, Run_H_KF_sparse_splitted, Run_H_KF_sparse  

To save the results
-------------------
.. automodule:: Run.General_analyses
   :members: SaveResultsKF
