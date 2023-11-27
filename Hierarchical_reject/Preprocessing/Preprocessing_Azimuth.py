## Packages
import h5py

import numpy as np
import pandas as pd

# import dask.dataframe as dd
from scipy.io import mmread
from scipy.sparse import csc_matrix, csr_matrix



## Functions Specific for PBMC
def ReadIn_PBMC(h5file):
    """Read in the data and labels from the Azimuth PBMC dataset

    Parameters
    ----------
    h5file : str
        Local path to the Azimuth PBMC h5file 

    Returns
    -------
    sparse_data
        Sparse matrix
    labels
        List
    """
    f = h5py.File(h5file, "r")
    data = f["raw"]["X"]["data"][:]
    indices = f["raw"]["X"]["indices"][:]
    indptr = f["raw"]["X"]["indptr"][:]  # (or memory profiler package)
    sparse_data = csr_matrix((data, indices, indptr), (161764, 20729))
    celltypes_l1 = f["obs"]["celltype.l1"][:]
    celltypes_l2 = f["obs"]["celltype.l2"][:]
    celltypes_l3 = f["obs"]["celltype.l3"][:]

    labels = [
        level_1.decode("utf-8")
        + ";"
        + level_2.decode("utf-8")
        + ";"
        + level_3.decode("utf-8")
        for level_1, level_2, level_3 in zip(celltypes_l1, celltypes_l2, celltypes_l3)
    ]
    return (sparse_data, labels)


def filter_condition(f, condition, data, labels):
    idx = [i.decode().split("_")[0] == condition for i in f["obs"]["orig.ident"][:]]
    labels = pd.DataFrame(labels).loc[idx, :]
    print(labels.iloc[:, 0].values.tolist())
    data = data[idx, :]
    return (data, labels.iloc[:, 0].values.tolist())


def ReadIn_PBMC_filter(h5file, condition):
    f = h5py.File(h5file, "r")
    data = f["raw"]["X"]["data"][:]
    indices = f["raw"]["X"]["indices"][:]
    indptr = f["raw"]["X"]["indptr"][:]  # (or memory profiler package)
    sparse_matrix = csr_matrix((data, indices, indptr), (161764, 20729))
    celltypes_l1 = f["obs"]["celltype.l1"][:]
    celltypes_l2 = f["obs"]["celltype.l2"][:]
    celltypes_l3 = f["obs"]["celltype.l3"][:]

    labels = [
        level_1.decode("utf-8")
        + ";"
        + level_2.decode("utf-8")
        + ";"
        + level_3.decode("utf-8")
        for level_1, level_2, level_3 in zip(celltypes_l1, celltypes_l2, celltypes_l3)
    ]

    # Filter on conditions
    sparse_matrix, labels = filter_condition(f, condition, sparse_matrix, labels)

    return (sparse_matrix, labels)


def ReadIn_PBMC_notraw(h5file):
    f = h5py.File(h5file, "r")
    data = f["X"]["data"][:]
    indices = f["X"]["indices"][:]
    indptr = f["X"]["indptr"][:]  # (or memory profiler package)
    sparse_matrix = csr_matrix((data, indices, indptr), (161764, 20729))
    celltypes_l1 = f["obs"]["celltype.l1"][:]
    celltypes_l2 = f["obs"]["celltype.l2"][:]
    celltypes_l3 = f["obs"]["celltype.l3"][:]

    labels = [
        level_1.decode("utf-8")
        + ";"
        + level_2.decode("utf-8")
        + ";"
        + level_3.decode("utf-8")
        for level_1, level_2, level_3 in zip(celltypes_l1, celltypes_l2, celltypes_l3)
    ]
    return (sparse_matrix, labels)


def Process_labels_PBMC(
    labels, data, Convert_Double_Names, Convert_first_level, Convert_Other
):
    """Processing of the PBMC labels to complete the hierarchy

    Parameters
    ----------
    labels : list
        original labels read in from h5file
    data :sparse matrix
        Azimuth PBMC data
    Convert_Double_Names : dict
        Conversion dictionary to remove double naming in the labels e.g. "NK;NK;NK_1" --> "NK;NK_1" 
    Convert_first_level : dict
        Conversion dictionary to add an extra first level to all the labels (e.g. Myeloid, Lymphoid)
    Convert_Other : dict
        Conversion dictionary to handle all the labels categorised as 'other' in the dataset

    Returns
    -------
    labels
        list
    data
        sparse matrix
    """
    labels = list(
        map(
            lambda x: Convert_Double_Names[x]
            if x in Convert_Double_Names.keys()
            else x,
            labels,
        )
    )
    labels = list(
        map(
            lambda x: (";").join(
                [Convert_first_level[x.split(";")[0]]] + x.split(";")[1:]
            )
            if x.split(";")[0] in Convert_first_level.keys()
            else x,
            labels,
        )
    )
    labels = list(
        map(lambda x: Convert_Other[x] if x in Convert_Other.keys() else x, labels)
    )

    # labels = list(filter(lambda x: x != "REMOVE", labels))
    ind_notREMOVE = np.array(labels) != "REMOVE"
    labels = np.array(labels)[ind_notREMOVE].tolist()
    data = data[ind_notREMOVE, :]

    ind_notINTER = np.array(labels) != "INTERMEDIATE"
    labels = np.array(labels)[ind_notINTER].tolist()
    data = data[ind_notINTER, :]

    labels = ["root;" + x for x in labels]
    return labels, data


def Preprocess_filter_Azimuth_PBMC(data, labels):
    """Filtering of cell populations with less than 10 classes

    Parameters
    ----------
    data : sparse matrix
        Azimuth PBMC sata 
    labels : list
        Converted labels

    Returns
    -------
    data: sparse matrix
    labels: list
    """
    freq_counts = pd.DataFrame(labels).value_counts()
    removed_classes = freq_counts.index.values[freq_counts < 10]  # numpy.ndarray
    rem_classes = [i[0] for i in removed_classes]
    Cells_To_Keep = [i not in rem_classes for i in labels]
    labels = pd.DataFrame(labels).loc[Cells_To_Keep, :]
    data = data[Cells_To_Keep, :]
    return (data, labels)


def Preprocessing_Azimuth_PBMC(h5file):
    """Preprocessing function for the Azimuth PBMC dataset. 
       The hierarchical information in the labels is completed and formatted for hierarchical classification. The intermediate and unspecified labels are removed and the cell populations with less than 10 members are discarded.

    Parameters
    ----------
    h5file : str 
        Local path to Azimuth PBMC dataset (hdf5 file)
        
    Returns
    ------
    Data: sparse matrix
    Labels: list
    """
    # Read in the data
    data_init, labels_init = ReadIn_PBMC(h5file)
    
    correct_first_level = {
        "B": "Lymphoid;B",
        "CD4 T": "Lymphoid;T;CD4 T",
        "CD8 T": "Lymphoid;T;CD8 T",
        "other T": "Lymphoid;T;other T",
        "DC": "Myeloid;DC",
        "Mono": "Myeloid;Mono",
        "NK": "Lymphoid;NK",
    }

    correct_Other = {
        "other;Doublet;Doublet": "REMOVE",
        "other;Eryth;Eryth": "MEGAkaryocyte Erocyte progenitor;Eryth",
        "other;HSPC;HSPC": "INTERMEDIATE",  # intermediate level (= root)
        "other;ILC;ILC": "INTERMEDIATE",  # Intermediate level (= Lymphoid)
        "other;Platelet;Platelet": "MEGAkaryocyte Erocyte progenitor;Megakaryocyte;Platelet",
    }

    correct_Double_names = {
        "B;Plasmablast;Plasmablast": "B;Plasmablast;Normal Plasmablast",
        "CD4 T;CD4 CTL;CD4 CTL": "CD4 T;CD4 CTL",
        "CD4 T;CD4 Naive;CD4 Naive": "CD4 T;CD4 Naive",
        "CD4 T;CD4 Proliferating;CD4 Proliferating": "CD4 T;CD4 Proliferating",
        "CD8 T;CD8 Naive;CD8 Naive": "CD8 T;CD8 Naive;CD8 Naive_1",
        "CD8 T;CD8 Proliferating;CD8 Proliferating": "CD8 T;CD8 Proliferating",
        "DC;pDC;pDC": "DC;pDC",
        "Mono;CD14 Mono;CD14 Mono": "Mono;CD14 Mono",
        "Mono;CD16 Mono;CD16 Mono": "Mono;CD16 Mono",
        "DC;cDC1;cDC1": "DC;cDC1",
        "NK;NK Proliferating;NK Proliferating": "NK;NK Proliferating",
        "NK;NK_CD56bright;NK_CD56bright": "NK;NK_CD56bright",
        "other T;MAIT;MAIT": "other T;MAIT",
        "NK;NK;NK_1": "NK;NK_1",
        "NK;NK;NK_2": "NK;NK_2",
        "NK;NK;NK_3": "NK;NK_3",
        "NK;NK;NK_4": "NK;NK_4",
        }
    
    # Correct the hierarchy
    labels_correct, data_correct =Process_labels_PBMC(labels_init, data_init, correct_Double_names, correct_first_level, correct_Other)
    
    # Filter cell populations
    Data, Labels = Preprocess_filter_Azimuth_PBMC(data_correct, labels_correct)

    return(Data, Labels)


    
    
    
correct_first_level = {
    "B": "Lymphoid;B",
    "CD4 T": "Lymphoid;T;CD4 T",
    "CD8 T": "Lymphoid;T;CD8 T",
    "other T": "Lymphoid;T;other T",
    "DC": "Myeloid;DC",
    "Mono": "Myeloid;Mono",
    "NK": "Lymphoid;NK",
}

correct_Other = {
    "other;Doublet;Doublet": "REMOVE",
    "other;Eryth;Eryth": "MEGAkaryocyte Erocyte progenitor;Eryth",
    "other;HSPC;HSPC": "INTERMEDIATE",  # intermediate level (= root)
    "other;ILC;ILC": "INTERMEDIATE",  # Intermediate level (= Lymphoid)
    "other;Platelet;Platelet": "MEGAkaryocyte Erocyte progenitor;Megakaryocyte;Platelet",
}

correct_Double_names = {
    "B;Plasmablast;Plasmablast": "B;Plasmablast;Normal Plasmablast",
    "CD4 T;CD4 CTL;CD4 CTL": "CD4 T;CD4 CTL",
    "CD4 T;CD4 Naive;CD4 Naive": "CD4 T;CD4 Naive",
    "CD4 T;CD4 Proliferating;CD4 Proliferating": "CD4 T;CD4 Proliferating",
    "CD8 T;CD8 Naive;CD8 Naive": "CD8 T;CD8 Naive;CD8 Naive_1",
    "CD8 T;CD8 Proliferating;CD8 Proliferating": "CD8 T;CD8 Proliferating",
    "DC;pDC;pDC": "DC;pDC",
    "Mono;CD14 Mono;CD14 Mono": "Mono;CD14 Mono",
    "Mono;CD16 Mono;CD16 Mono": "Mono;CD16 Mono",
    "DC;cDC1;cDC1": "DC;cDC1",
    "NK;NK Proliferating;NK Proliferating": "NK;NK Proliferating",
    "NK;NK_CD56bright;NK_CD56bright": "NK;NK_CD56bright",
    "other T;MAIT;MAIT": "other T;MAIT",
    "NK;NK;NK_1": "NK;NK_1",
    "NK;NK;NK_2": "NK;NK_2",
    "NK;NK;NK_3": "NK;NK_3",
    "NK;NK;NK_4": "NK;NK_4",
}
