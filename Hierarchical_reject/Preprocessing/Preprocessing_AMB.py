# Preprocessing methods

## Packages
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from sklearn.preprocessing import StandardScaler


def Preprocessing_AMB(DataPath, LabelPath):
    """Preprocessing function for the Allen Mouse Brain dataset
       Cell populations with less than 10 members are filtered out and the labels are converted to the correct format for hierarchical classification.
    Parameters
    ----------
    DataPath : str
        Local path to the COVID dataset
    LabelPath : str
        Local path to labels of COVID dataset

    Returns
    -------
    Data
        Pandas dataframe
    Labels
        List
    """
    Data_init = pd.read_csv(DataPath, index_col=0, sep=",")
    Labels_init = pd.read_csv(LabelPath, header=0, index_col=None, sep=",")

    # filtering
    l2 = Labels_init["cluster"].value_counts()
    removed_classes = l2.index.values[l2 < 10]  # numpy.ndarray
    Cells_To_Keep = [
        i
        for i in range(len(Labels_init["cluster"]))
        if not Labels_init["cluster"][i] in removed_classes
    ]  # list with indices
    labels = Labels_init.iloc[Cells_To_Keep]
    Data = Data_init.iloc[Cells_To_Keep]

    labels_newFormat = []
    for index, row in labels.iterrows():
        line = "root;" + row.Class + ";" + row.Subclass + ";" + row.cluster
        labels_newFormat.append(line)
    Labels= pd.DataFrame(labels_newFormat)
    return (Data, Labels)
