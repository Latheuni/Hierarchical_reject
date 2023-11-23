import pandas as pd
import numpy as np


def PreprocessingCOVID(
    LabelPath, DataPath, filter_prolif=True, filter_unspecified=False
):
    """
    Parameters
    ----------
    LabelPath : str
        Local Path to Labels of COVID dataset
    DataPath : str
        Local Path to Labels of COVID dataset
    filter_prolif : Boolean optional
        Filter out the Proliferating cell state labels?. The default is True.
    filter_unspecified : Boolean, optional
        Filter out the undpecified labels?. The default is False.

    Returns
    -------
    Data: filtered data dataframe
    Labels: with in column 3 the input data for hclf

    """
    Labels = pd.read_csv(LabelPath, header=None, index_col=None, sep=",")
    Data = pd.read_csv(DataPath, sep=",").T

    # Filter QC, Undefined and prolif if neccessary
    classes_to_filter = ("QC", "Undefined")

    if filter_prolif == True:
        classes_to_filter = classes_to_filter + ("Prolif",)

    classes_to_filter_index = ~Labels[1].str.startswith(classes_to_filter).values
    Labels = Labels.iloc[classes_to_filter_index, :]
    Data = Data.iloc[classes_to_filter_index, :]

    # Filter TOP 10 classes
    freq_counts = Labels[1].value_counts()
    removed_classes = freq_counts.index.values[freq_counts < 10]  # numpy.ndarray

    Cells_To_Keep = [i not in removed_classes for i in Labels[1]]
    Data = Data.iloc[Cells_To_Keep, :]
    Labels = Labels.iloc[Cells_To_Keep, :]

    # Change names where necessary
    d_convertLabels = {}
    d_convertLabels["Epi_Secretory_Mucous"] = "Epi_Secretory-Mucous"
    d_convertLabels["L_T_CD8_activ"] = "L_T_CD8_activ-CD8"
    d_convertLabels["Epi_AT_AT2_stress"] = "Epi_AT_AT2"
    if filter_unspecified == False:
        d_convertLabels["Epi_Unspecified_Unspec-Epi"] = "Epi_Unspec-Epi"
        d_convertLabels["L_T_CD4_Unspecified_Unspec-CD4-1"] = "L_T_CD4_Unspec-CD41"
        d_convertLabels["L_T_CD4_Unspecified_Unspec-CD4-2"] = "L_T_CD4_Unspec-CD42"
        d_convertLabels["Prolif_Unidentified_Unident-Prolif"] = "Prolif_Unident-Polif"
    else:
        d_convertLabels["L_T_CD8_Unspec-CD8"] = "L_T_CD8_Unspecified_Unspec-CD8"
        d_convertLabels[
            "Prolif_Unidentified_Unident-Prolif"
        ] = "Prolif_Unspecified_Unident-Polif"

    if filter_prolif == False:
        d_convertLabels["Prolif_L_NK"] = "Prolif_Prolif-L_Prolif-NK"
        d_convertLabels["Prolif_L_T_CD4"] = "Prolif_Prolif-L_Prolif-T_Prolif-CD4"
        d_convertLabels["Prolif_L_T_CD4_Treg"] = "Prolif_Prolif-L_Prolif-T_Prolif-CD4"
        d_convertLabels["Prolif_L_T_CD8"] = "Prolif_Prolif-L_Prolif-T_Prolif-CD8"
        d_convertLabels[
            "Prolif_M_Macrophage_Alveolar"
        ] = "Prolif_Prolif-M_Prolif-Macrophage_Prolif-Alveolar"

    labels_converted = [
        d_convertLabels[i] if i in d_convertLabels.keys() else i for i in Labels[1]
    ]
    Labels.insert(2, "Labels_converted_names", labels_converted)

    # Filter unspecified if necessary
    if filter_unspecified == True:
        filtered_classes_unspecified = [
            "Unspecified" not in i.split("_") for i in Labels
        ]
        Data = Data.iloc[filtered_classes_unspecified, :]
        Labels = Labels.iloc[filtered_classes_unspecified, :]

    # Reformat for hclf
    labels_correct_format = [
        "root;" + ";".join(i.split("_")) for i in Labels["Labels_converted_names"]
    ]
    Labels.insert(3, "Labels hclf format", labels_correct_format)

    return (Data, Labels)
