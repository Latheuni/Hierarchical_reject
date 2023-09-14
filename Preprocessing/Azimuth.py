## Packages
import h5py
import time
import numpy as np
import pandas as pd

# import dask.dataframe as dd
from scipy.io import mmread
from scipy.sparse import csc_matrix, csr_matrix

## Functions Specific for Motor Cortex
def ReadInMetadata(metadatafile):
    df = pd.read_csv(metadatafile)
    labels = [
        "root;" + level_1 + ";" + level_2 + ";" + level_3
        for level_1, level_2, level_3 in zip(
            df["class_label"], df["subclass_label"], df["cluster_label"]
        )
    ]
    samples = df["sample_name"].to_list()
    conversion_Table = pd.DataFrame([samples, labels])
    conversion_Table = conversion_Table.T
    conversion_Table.columns = [
        "sample_name",
        "labels",
    ]
    return conversion_Table


def ReadInData(datafile, metadatafile):
    start = time.time()
    ddf = pd.read_csv(datafile, sample=10000000)
    end = time.time()
    print("Loaded in gene expression matrix in", start - end, "seconds")
    print("shape dask dataframe", ddf.shape)
    samples = ddf["sample_name"]
    conversion_Table = ReadInMetadata(metadatafile)
    print("shape conversion table", conversion_Table.shape)
    df = ddf.merge(conversion_Table, on="sample_name")
    print("Shape merged dataframe", df.shape)
    print("columns merged dataframe", df.columns)
    return df  # Contains sample_names and labels, be carefull


def PreprocessAzimuthCortex(df):
    labels = list(df["labels"])
    data = df.loc[:, ~df.columns.isin(["sample_name", "labels"])]
    print(data.memory_usage(deep=True).sum())
    print(data.memory_usage(deep=True).sum().compute())
    # data = data.compute()
    print("shape data", data.shape)

    freq_counts = pd.DataFrame(labels).iloc[:, 0].value_counts()
    removed_classes = freq_counts.index.values[freq_counts < 10]  # numpy.ndarray
    print("removed classes", removed_classes)
    Cells_To_Keep = [i not in removed_classes for i in labels]
    labels = pd.DataFrame(labels).loc[Cells_To_Keep, :].iloc[:, 0].tolist()
    data = data.loc[Cells_To_Keep, :]
    return (data, labels)


## Functions Specific for Kidney
def ReadInMetadataKidney(csvfile, correct_labels_Kidney):
    df = pd.read_csv(csvfile)
    labels = [
        level_1 + ";" + level_2 + ";" + level_3
        for level_1, level_2, level_3 in zip(
            df["annotation.l1"], df["annotation.l2"], df["annotation.l3"]
        )
    ]  # assume data amtrix correct order (ori.index)
    print(len(labels))
    labels = list(
        map(
            lambda x: "root;" + correct_labels_Kidney[x]
            if x in correct_labels_Kidney.keys()
            else "root;" + x,
            labels,
        )
    )
    print(len(labels))
    return labels


def ReadAndPreProcess_Kidney(mtxfile, csvfile, correct_labels_Kidney):
    data_sparse = mmread(mtxfile)
    data = data_sparse.tocsr().T
    labels = ReadInMetadataKidney(csvfile, correct_labels_Kidney)

    print(len(labels))
    freq_counts = pd.DataFrame(labels).value_counts()
    removed_classes = freq_counts.index.values[freq_counts < 10]
    rem_classes = [i[0] for i in removed_classes]  # numpy.ndarray
    print("removed classes", rem_classes)
    Cells_To_Keep = [i not in removed_classes for i in labels]
    labels = pd.DataFrame(labels).loc[Cells_To_Keep, :]
    print(len(labels))
    data = data[Cells_To_Keep, :]
    print(data.shape)
    return (data, labels)


correct_labels_Kidney = {
    "Ascending Thin Limb;Ascending Thin Limb;Ascending Thin Limb": "Thin Limb;Ascending Thin Limb",
    "Descending Thin Limb;Descending Thin Limb Type 1;Descending Thin Limb Type 1": "Thin Limb;Descending Thin Limb;Descending Thin Limb Type 1",
    "Descending Thin Limb;Descending Thin Limb Type 2;Descending Thin Limb Type 2": "Thin Limb;Descending Thin Limb;Descending Thin Limb Type 2",
    "Descending Thin Limb;Descending Thin Limb Type 3;Descending Thin Limb Type 3": "Thin Limb;Descending Thin Limb;Descending Thin Limb Type 3",
    "Connecting Tubule;Connecting Tubule;Connecting Tubule": "Tubule;Connecting Tubule; Connecting Tubule normal",
    "Connecting Tubule;Connecting Tubule;Connecting Tubule Principal": "Tubule;Connecting Tubule;Connecting Tubule Principal",
    "Distal Convoluted Tubule;Distal Convoluted Tubule;Distal Convoluted Tubule Type 1": "Tubule;Distal Convoluted Tubule;Distal Convoluted Tubule Type 1",
    "Distal Convoluted Tubule;Distal Convoluted Tubule;Distal Convoluted Tubule Type 2": "Tubule;Distal Convoluted Tubule;Distal Convoluted Tubule Type 2",
    "Endothelial;Afferent / Efferent Arteriole Endothelial;Afferent / Efferent Arteriole Endothelial": "Endothelial;Afferent / Efferent Arteriole Endothelial",
    "Endothelial;Ascending Vasa Recta Endothelial;Ascending Vasa Recta Endothelial": "Endothelial;Ascending Vasa Recta Endothelial",
    "Endothelial;Descending Vasa Recta Endothelial ;Descending Vasa Recta Endothelial ": "Endothelial;Descending Vasa Recta Endothelial",
    "Endothelial;Glomerular Capillary Endothelial;Glomerular Capillary Endothelial": "Endothelial;Glomerular Capillary Endothelial",
    "Endothelial;Lymphatic Endothelial;Lymphatic Endothelial": "Endothelial;Lymphatic Endothelial",
    "Endothelial;Peritubular Capilary Endothelial ;Peritubular Capilary Endothelial ": "Endothelial;Peritubular Capilary Endothelial",
    "Fibroblast;Fibroblast;Fibroblast": "Fibroblast;Normal Fibroblast",
    "Fibroblast;Medullary Fibroblast;Medullary Fibroblast": "Fibroblast;Medullary Fibroblast",
    "Immune;B;B": "Immune;B",
    "Immune;Classical Dendritic;Classical Dendritic": "Immune;Classical Dendritic",
    "Immune;Cycling Mononuclear Phagocyte;Cycling Mononuclear Phagocyte": "Immune;Cycling Mononuclear Phagocyte",
    "Immune;M2 Macrophage;M2 Macrophage": "Immune;M2 Macrophage",
    "Immune;Mast;Mast": "Immune;Mast",
    "Immune;Monocyte-derived;Monocyte-derived": "Immune;Monocyte-derived",
    "Immune;Natural Killer T;Natural Killer T": "Immune;Natural Killer T",
    "Immune;Neutrophil;Neutrophil": "Immune;Neutrophil",
    "Immune;Non-classical monocyte;Non-classical monocyte": "Immune;Non-classical monocyte",
    "Immune;Plasma;Plasma": "Immune;Plasma",
    "Immune;Plasmacytoid Dendritic;Plasmacytoid Dendritic": "Immune;Plasmacytoid Dendritic",
    "Immune;T;T": "Immune;T",
    "Intercalated;Cortical Collecting Duct Intercalated Type A;Cortical Collecting Duct Intercalated Type A": "Intercalated;Cortical Collecting Duct Intercalated Type A;Normal Cortical Collecting Duct Intercalated Type A",
    "Intercalated;Intercalated Type B;Intercalated Type B": "Intercalated;Intercalated Type B",
    "Intercalated;Outer Medullary Collecting Duct Intercalated Type A;Outer Medullary Collecting Duct Intercalated Type A": "Intercalated;Outer Medullary Collecting Duct Intercalated Type A",
    "Papillary Tip Epithelial;Papillary Tip Epithelial;Papillary Tip Epithelial": "Papillary Tip Epithelial",
    "Parietal Epithelial;Parietal Epithelial;Parietal Epithelial": "Parietal Epithelial",
    "Podocyte;Podocyte;Podocyte": "Podocyte",
    "Principal;Cortical Collecting Duct Principal;Cortical Collecting Duct Principal": "Principal;Cortical Collecting Duct Principal",
    "Principal;Inner Medullary Collecting Duct;Inner Medullary Collecting Duct": "Principal;Inner Medullary Collecting Duct",
    "Principal;Outer Medullary Collecting Duct Principal;Outer Medullary Collecting Duct Principal": "Principal;Outer Medullary Collecting Duct Principal",
    "Proximal Tubule;Proximal Tubule Epithelial Segment 1;Proximal Tubule Epithelial Segment 1": "Proximal Tubule;Proximal Tubule Epithelial Segment 1",
    "Proximal Tubule;Proximal Tubule Epithelial Segment 2;Proximal Tubule Epithelial Segment 2": "Proximal Tubule;Proximal Tubule Epithelial Segment 2",
    "Proximal Tubule;Proximal Tubule Epithelial Segment 3;Proximal Tubule Epithelial Segment 3": "Proximal Tubule;Proximal Tubule Epithelial Segment 3",
    "Schwann;Schwann / Neural;Schwann / Neural": "Schwann;Schwann / Neural",
    "Thick Ascending Limb;Cortical Thick Ascending Limb;Cortical Thick Ascending Limb": "Thick Ascending Limb;Cortical Thick Ascending Limb",
    "Thick Ascending Limb;Macula Densa;Macula Densa": "Thick Ascending Limb;Macula Densa",
    "Thick Ascending Limb;Medullary Thick Ascending Limb;Medullary Thick Ascending Limb": "Thick Ascending Limb;Medullary Thick Ascending Limb",
    "Vascular Smooth Muscle / Pericyte;Cortical Vascular Smooth Muscle / Pericyte;Cortical Vascular Smooth Muscle / Pericyte": "Vascular Smooth Muscle / Pericyte;Cortical Vascular Smooth Muscle / Pericyte",
    "Vascular Smooth Muscle / Pericyte;Mesangial;Mesangial": "Vascular Smooth Muscle / Pericyte;Mesangial",
    "Vascular Smooth Muscle / Pericyte;Renin-positive Juxtaglomerular Granular;Renin-positive Juxtaglomerular Granular": "Vascular Smooth Muscle / Pericyte;Renin-positive Juxtaglomerular Granular",
    "Vascular Smooth Muscle / Pericyte;Vascular Smooth Muscle / Pericyte;Vascular Smooth Muscle / Pericyte": "Vascular Smooth Muscle / Pericyte;Normal Vascular Smooth Muscle / Pericyte",
}

## Functions Specific for PBMC
def ReadIn_PBMC(h5file):
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
    return (sparse_matrix, labels)


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
    labels = list(
        map(
            lambda x: Convert_Double_Names[x]
            if x in Convert_Double_Names.keys()
            else x,
            labels,
        )
    )
    print(np.unique(labels))
    print(len(labels))
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

    print(np.unique(labels))
    print(len(labels))
    print(data.shape)
    return labels, data


def Preprocess_PBMC(data, labels):
    freq_counts = pd.DataFrame(labels).value_counts()
    removed_classes = freq_counts.index.values[freq_counts < 10]  # numpy.ndarray
    rem_classes = [i[0] for i in removed_classes]
    print("removed classes", removed_classes)
    Cells_To_Keep = [i not in rem_classes for i in labels]
    print(sum(Cells_To_Keep))
    labels = pd.DataFrame(labels).loc[Cells_To_Keep, :]
    print(len(labels))
    data = data[Cells_To_Keep, :]
    return (data, labels)


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
