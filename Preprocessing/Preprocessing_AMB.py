# Preprocessing methods

## Packages
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from sklearn.preprocessing import StandardScaler


def PreprocessingAMB(DataPath, LabelPath):
    Data = pd.read_csv(DataPath, index_col=0, sep=",")
    Labels = pd.read_csv(LabelPath, header=0, index_col=None, sep=",")

    # filtering
    l2 = Labels["cluster"].value_counts()
    removed_classes = l2.index.values[l2 < 10]  # numpy.ndarray
    Cells_To_Keep = [
        i
        for i in range(len(Labels["cluster"]))
        if not Labels["cluster"][i] in removed_classes
    ]  # list with indices
    labels = Labels.iloc[Cells_To_Keep]
    data = Data.iloc[Cells_To_Keep]

    labels_newFormat = []
    for index, row in labels.iterrows():
        line = "root;" + row.Class + ";" + row.Subclass + ";" + row.cluster
        labels_newFormat.append(line)
    Labels_newFormat = pd.DataFrame(labels_newFormat)
    return (data, Labels_newFormat)
