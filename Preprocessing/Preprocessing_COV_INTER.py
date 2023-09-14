# Preprocessing_Functions COV_INTER
## Packages

import os
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix


def Input_COVID(filename):
    f = h5py.File(filename, "r")

    data = f["RNA"]["raw_counts"]["data"][:]
    indices = f["RNA"]["raw_counts"]["indices"][
        :
    ]  # Is for the columns length is the same as all the non-zero entries
    indptr = f["RNA"]["raw_counts"]["indptr"][
        :
    ]  # Is for the rows: contains all the cells

    data_sparse = csc_matrix((data, indices, indptr), shape=(24444, 275056))

    labels = [str(i)[2:-1] for i in f["metadata"]["label"]]
    patients = [str(i)[2:-1] for i in f["metadata"]["orig.ident"]]
    covid = [str(i)[2:-1] for i in f["metadata"]["covid19"]]

    ds = pd.DataFrame([labels, patients, covid]).T
    ds.columns = ["Label", "Patient", "Disease"]
    return (data_sparse, ds)


def GetDiseaseData(data_sparse, ds, cov="POS"):
    data_cov_sparse = data_sparse[
        :, ds["Disease"] == cov
    ]  # csc so I am selecting columns now: ok
    # print(data_cov_sparse.get_shape()) # (24444,140014)

    if data_cov_sparse.get_shape()[1] != sum(ds["Disease"] == cov):
        raise Exception("Something went wrong with formatting Data")

    labels_cov = ds[ds["Disease"] == cov]
    patients_cov = ds[ds["Disease"] == cov]["Patient"]
    if len(labels_cov) != sum(ds["Disease"] == cov):
        raise Exception("Something went wrong with formatting Labels")
    return (data_cov_sparse, labels_cov, patients_cov)


def reformatLabelsCOVIDFull(data, Labels, filter_prolif=True, filter_unspec=True):
    # print('type data', type(data)) #is a sparse matrix
    Data = data.transpose()
    # print("length labels", len(Labels)) # 140 014
    # print(Data.get_shape()) #(140014, 24444)
    # print('type labels', type(Labels)) # is a Series
    Labels = pd.DataFrame(Labels)
    # print(Labels)
    if filter_unspec == True:
        filtered_classes_unspecified = [
            "Unspecified" not in i.split("_") for i in Labels["Label"]
        ]

        Data = Data[np.array(filtered_classes_unspecified).nonzero()[0], :]
        # print(
        #    "Unspecified labels",
        #    np.unique(Labels.iloc[filtered_classes_unspecified, :]),
        # )
        Labels = Labels.iloc[filtered_classes_unspecified, :]
        print("Data filter 1", Data.get_shape())
        print("Labels filter 1", Labels.shape)
        print("\n")

        filtered_classes_undefined = [
            "Undefined" not in i.split("_") for i in Labels["Label"]
        ]

        Data = Data[np.array(filtered_classes_undefined).nonzero()[0], :]
        # print("Undefined labels", np.unique(Labels.iloc[filtered_classes_undefined, :]))
        Labels = Labels.iloc[filtered_classes_undefined, :]
        print("Data filter 2", Data.get_shape())
        print("Labels filter 2", Labels.shape)
        print("\n")

        filtered_classes_unidentified = [
            "Unidentified" not in i.split("_") for i in Labels["Label"]
        ]

        Data = Data[np.array(filtered_classes_unidentified).nonzero()[0], :]
        # print(
        #    "Undefined labels", np.unique(Labels.iloc[filtered_classes_unidentified, :])
        # )

        Labels = Labels.iloc[filtered_classes_unidentified, :]
        print("Data filter 3", Data.get_shape())
        print("Labels filter 3", Labels.shape)
        print("\n")

    # filter QCs
    noQCs = ~Labels["Label"].str.startswith("QC").values
    print("N° of QCs", sum(noQCs))
    print("\n")

    Data = Data[np.array(noQCs).nonzero()[0], :]
    Labels = Labels.iloc[noQCs, :]
    print("Data filter 4", Data.get_shape())
    print("Labels filter 4", Labels.shape)
    print("\n")

    # Reformat labels based on dict
    d_convertLabels = {}
    ## The names of the nodes need to be unique or a DAG will be formed

    d_convertLabels["L_T_CD8_activ"] = "L_T_CD8_activ-CD8"
    d_convertLabels["L_T_CD4_activ"] = "L_T_CD4_activ-CD4"
    d_convertLabels["Epi_AT_AT2_stress"] = "Epi_AT_AT2"
    d_convertLabels[
        "M_Macrophage_recruited-activated"
    ] = "M_Macrophage_recruited-activated-Macro"
    d_convertLabels[
        "M_Monocyte_recruited-activated"
    ] = "M_Monocyte_recruited-activated-Mono"
    d_convertLabels["M_Macrophage_patrolling"] = "M_Macrophage_patrolling-Macro"
    d_convertLabels["M_Monocyte_patrolling"] = "M_Monocyte_patrolling-Mono"
    if filter_unspec == False:
        d_convertLabels["Epi_Unspecified_Unspec-Epi"] = "Epi_Unspec-Epi"
        d_convertLabels["L_T_CD4_Unspecified_Unspec-CD4-1"] = "L_T_CD4_Unspec-CD41"
        d_convertLabels["L_T_CD4_Unspecified_Unspec-CD4-2"] = "L_T_CD4_Unspec-CD42"
        d_convertLabels["Prolif_Unidentified_Unident-Prolif"] = "Prolif_Unident-Polif"
        d_convertLabels["Undefined_Undefined-L"] = "Undefined-L"
        d_convertLabels["L_T_CD8_Unspecified_Unspec-CD8"] = "L_T_CD8_Unspec-CD8"
    if filter_prolif == False:
        d_convertLabels["Prolif_L_NK"] = "Prolif_Prolif-L_Prolif-NK"
        d_convertLabels["Prolif_L_T_CD4"] = "Prolif_Prolif-L_Prolif-T_Prolif-CD4"
        d_convertLabels["Prolif_L_T_CD4_Treg"] = "Prolif_Prolif-L_Prolif-T_Prolif-CD4"
        d_convertLabels["Prolif_L_T_CD8"] = "Prolif_Prolif-L_Prolif-T_Prolif-CD8"
        d_convertLabels[
            "Prolif_M_Macrophage_Alveolar"
        ] = "Prolif_Prolif-M_Prolif-Macrophage_Prolif-Alveolar"
    Labels_list = [
        d_convertLabels[i] if i in d_convertLabels.keys() else i
        for i in Labels["Label"]
    ]

    # Filter prolif if True:
    if filter_prolif == True:
        noProlifs = ~np.array([i.startswith("Prolif") for i in Labels_list])
        Data = Data[~np.array(noProlifs).nonzero()[0], :]
        Labels = Labels.loc[noProlifs, :]
        Labels_list = np.array(Labels_list)[noProlifs]  # LABELS IS NUMPY ARRAY
        print("Data filter 5", Data.get_shape())
        print("labels list filter 5", len(Labels_list))
        print("labelsfilter 5", Labels.shape)
        print("\n")

    freq_counts = pd.DataFrame(Labels_list)[0].value_counts()
    removed_classes = freq_counts.index.values[freq_counts < 10]  # numpy.ndarray
    Cells_To_Keep = [i not in removed_classes for i in Labels_list]
    Data = Data[np.array(Cells_To_Keep).nonzero()[0], :]
    Labels = Labels.loc[Cells_To_Keep, :]
    Labels_list = list(Labels_list[Cells_To_Keep])
    print("Data filter 6", Data.get_shape())
    print("labelsfilter 6", Labels.shape)
    print("\n")

    # Reformat for hclf
    labels_correct_format = ["root;" + ";".join(i.split("_")) for i in Labels_list]
    Labels.insert(1, "labels_correct_format", labels_correct_format)
    print("reformat data", Data.get_shape())
    print("reformat labels", Labels.shape)
    print("reformat unique labels", np.unique(Labels["labels_correct_format"]))
    print("\n")
    return (Data, Labels)


def reformat_patients(data, labels):
    selected_patients = ["COV012", "COV013", "COV024", "COV034"]
    filter_patients = [patient in selected_patients for patient in labels["Patient"]]

    labels = labels.loc[filter_patients, :]
    print("represented patients", np.unique(labels["Patient"]))
    data = data[np.array(filter_patients).nonzero()[0], :]
    print("select patients data", data.get_shape())
    print("select patients labels", labels.shape)
    print("\n")

    to_filter_labels = [
        "root;G;Basophil",
        "root;Epi;Secretory;Serous",
        "root;Epi;CD309",
        "root;Epi;Basal",
        "root;Epi;AT;AT1",
        "root;Epi;AT;AT2",
    ]
    labels_filter = [
        label not in to_filter_labels for label in labels["labels_correct_format"]
    ]
    labels = labels.loc[labels_filter, :]
    data = data[np.array(labels_filter).nonzero()[0], :]
    print("reformat patients data", data.get_shape())
    print("reformat patients labels", labels.shape)
    print(
        "reformat patients unique labels",
        len(np.unique(labels["labels_correct_format"])),
    )
    print("\n")
    return (data, labels)


def reformat_patients2(data, labels):
    print(labels["Patient"][0])
    selected_patients = ["COV012", "COV013"]
    filter_patient1 = [
        labels["Patient"].iloc[i] == selected_patients[0]
        for i in range(0, len(labels["Patient"]))
    ]
    filter_patient11 = [
        True if filter_patient1[i] == True and sum(filter_patient1[0:i]) < 50 else False
        for i in range(0, len(filter_patient1))
    ]
    print(sum(filter_patient1))
    print(sum(filter_patient11))
    filter_patient2 = [
        labels["Patient"].iloc[i] == selected_patients[1]
        for i in range(0, len(labels["Patient"]))
    ]
    filter_patient22 = [
        True if filter_patient2[i] == True and sum(filter_patient2[0:i]) < 50 else False
        for i in range(0, len(filter_patient2))
    ]
    print(sum(filter_patient2))
    print(sum(filter_patient22))
    filter_patients = np.logical_or(filter_patient11, filter_patient22)
    print(sum(filter_patients))
    labels = labels.loc[filter_patients, :]
    data = data[np.array(filter_patients).nonzero()[0], :]
    print("select patients data", data.get_shape())
    print("select patients labels", labels.shape)
    print("\n")

    to_filter_labels = [
        "root;G;Basophil",
        "root;Epi;Secretory;Serous",
        "root;Epi;CD309",
        "root;Epi;Basal",
        "root;Epi;AT;AT1",
        "root;Epi;AT;AT2",
    ]
    labels_filter = [
        label not in to_filter_labels for label in labels["labels_correct_format"]
    ]
    labels = labels.loc[labels_filter, :]
    data = data[np.array(labels_filter).nonzero()[0], :]
    print("reformat patients data", data.get_shape())
    print("reformat patients labels", labels.shape)
    print(
        "reformat patients unique labels",
        len(np.unique(labels["labels_correct_format"])),
    )
    print("\n")
    return (data, labels)


def reformat_patientsPOS(data, labels):
    print(labels.head())
    selected_patients = [
        "COV002",
        "COV012",
        "COV013",
        "COV015",
        "COV024",
        "COV034",
        "COV036",
        "COV037",
    ]
    unique_labels = []
    filter = [p in selected_patients for p in labels["Patient"]]
    print("amount of trues in filter", sum(filter))
    for patient in selected_patients:
        filter_patients = [p == patient for p in labels["Patient"]]
        unique_labels.append(
            list(np.unique(labels.loc[filter_patients]["labels_correct_format"]))
        )

    common_labels = set(unique_labels[0])
    for s in unique_labels[1:]:
        common_labels.intersection_update(s)

    labels = labels.loc[filter, :]
    data = data[np.array(filter).nonzero()[0], :]
    print("select patients data", data.get_shape())
    print("select patients labels", labels.shape)
    print("\n")

    labels_filter = [
        label in common_labels for label in labels["labels_correct_format"]
    ]
    labels = labels.loc[labels_filter, :]
    data = data[np.array(labels_filter).nonzero()[0], :]
    print("reformat patients data", data.get_shape())
    print("reformat patients labels", labels.shape)
    print(
        "reformat patients unique labels",
        len(np.unique(labels["labels_correct_format"])),
    )
    print("\n")
    return (data, labels)


def reformat_patientsNEG(data, labels):
    print(labels.head())
    selected_patients = [
        "COV004",
        "COV006",
        "COV007",
        "COV014",
        "COV021",
        "COV022",
        "COV023",
        "COV025",
        "COV035",
    ]
    unique_labels = []
    filter = [p in selected_patients for p in labels["Patient"]]
    print("amount of trues in filter", sum(filter))
    for patient in selected_patients:
        filter_patients = [p == patient for p in labels["Patient"]]
        unique_labels.append(
            list(np.unique(labels.loc[filter_patients]["labels_correct_format"]))
        )

    common_labels = set(unique_labels[0])
    for s in unique_labels[1:]:
        common_labels.intersection_update(s)

    labels = labels.loc[filter, :]
    data = data[np.array(filter).nonzero()[0], :]
    print("select patients data", data.get_shape())
    print("select patients labels", labels.shape)
    print("\n")

    labels_filter = [
        label in common_labels for label in labels["labels_correct_format"]
    ]
    labels = labels.loc[labels_filter, :]
    data = data[np.array(labels_filter).nonzero()[0], :]
    print("reformat patients data", data.get_shape())
    print("reformat patients labels", labels.shape)
    print(
        "reformat patients unique labels",
        len(np.unique(labels["labels_correct_format"])),
    )
    print("\n")
    return (data, labels)
