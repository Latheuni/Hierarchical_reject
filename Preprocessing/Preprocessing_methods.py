# Preprocessing methods

## Packages
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from sklearn.preprocessing import StandardScaler

## Functions
class NormAndScaleMethods:
    def __init__(self, data):
        adata = AnnData(data)
        adata.raw = adata
        self.adata = adata

    def Scale_and_Clip(self, m=10):
        # scale each gene to unit variance, clip values exceeding standard deviation 10
        d = sc.pp.scale(self.adata.raw.X, max_value=m, copy=True)
        return d

    def Scale_fit(self, train_data):
        # standardized across the feature axis
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_data)
        self.model = scaler
        return train_scaled

    def Scale_transform(self, test_data):
        m = self.model
        return m.transform(test_data)

    def Library_size_correct(self, sum=1e6):  # CPM normalization
        d = sc.pp.normalize_total(self.adata, target_sum=sum, copy=True)
        return d.X


class HVGselection:
    def __init__(self, flavour, top_genes=1000):
        self.flavour = flavour
        self.top_genes = top_genes

    def fit(
        self, data, labels
    ):  # don't need labels here but has to do with Pipeline construction
        adata = AnnData(data)
        # sc.pp.normalize_total(adata) doesn't work with normalize after log
        # sc.pp.log1p(adata): already done if normalize = True
        sc.pp.highly_variable_genes(
            adata, n_top_genes=self.top_genes, flavor=self.flavour
        )
        self.highly_variable_genes = adata.var["highly_variable"]  # True/False list
        return self

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def transform(self, data):
        if isinstance(data, pd.DataFrame):
            return data.loc[:, self.highly_variable_genes]
        else:
            return data[:, self.highly_variable_genes]


def F_test(X, y):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    unique_classes = np.unique(y)

    row_class_mean = np.zeros((len(X), len(unique_classes)))
    row_means = X.mean(axis=1)  # output is an numpy array, input should be a dataframe
    for i in range(0, len(unique_classes)):
        ix = np.where(np.array(y) == unique_classes[i])[0]
        if ix.shape[0] > 1:
            row_class_mean[:, i] = X.iloc[:, ix].mean(axis=1).values  # rowmeans
        else:
            row_class_mean[:, i] = X.iloc[:, ix[0]].values

    row_class_mean_df = pd.DataFrame(row_class_mean, columns=unique_classes)

    freq_counts = pd.DataFrame(y).value_counts()[unique_classes].values
    # table_class = table(classes)[unique_classes]
    BBS = np.matmul(
        freq_counts,
        np.transpose((row_class_mean_df - np.atleast_2d(row_means).T) ** 2).values,
    )
    # TSS = pd.DataFrame((X - row_means[:,np.newaxis])**2).sum(axis=1)
    TSS = pd.DataFrame((X - np.atleast_2d(row_means).T) ** 2).sum(axis=1)
    ESS = TSS - BBS
    df1 = len(unique_classes) - 1
    df2 = len(X.columns) - len(unique_classes)
    Fscore = (BBS / df1) / (ESS / df2)
    # crit_value = scipy.stat.ppf(q=1-0.05, dfn = df1, dfd = df2) # With this I think everything larger than this will have a 95% confidence
    return Fscore


class Fselection:
    """
    Normalise beforehand! (Is in assumption!)
    """

    def __init__(self, n_features=1000):
        self.n_features = n_features

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(
        self, data, labels
    ):  # don't need labels here but has to do with Pipeline construction
        F_scores = F_test(data.T, labels)
        idx = np.argsort(-(F_scores.values))
        # print(F_scores.values[idx])
        # print(idx)
        self.selected_features = idx[0 : self.n_features]
        return self

    def transform(self, data):
        if isinstance(data, pd.DataFrame):
            return data.iloc[:, self.selected_features]
        else:
            return data[:, self.selected_features]


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
