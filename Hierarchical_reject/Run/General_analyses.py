########### Packages ###########
import os
import csv
import scipy.sparse
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    ParameterGrid,
    GroupKFold,
)
from sklearn.base import clone
from hclf.multiclass import LCPN  # SHOULD BE MULTICLASS_NEW ON HPC!!
import scanpy as sc
from anndata import AnnData

########### Utility function ###########
def predict_proba_from_scores(estimator, X):
    # Based on the code thomas wrote
    # get scores
    scores = estimator.decision_function(X)
    scores = np.exp(scores)
    # check if we only have one score (ie, when K=2)
    if len(scores.shape) == 2:
        # softmax evaluation
        scores = scores / np.sum(scores, axis=1).reshape(scores.shape[0], 1)
    else:
        # sigmoid evaluation
        scores = 1 / (1 + np.exp(-scores))
        scores = scores.reshape(-1, 1)
        scores = np.hstack([1 - scores, scores])
    return scores
########### Feature Selection ###########
def F_test_sparse(X, y):
    unique_classes = np.unique(y)
    row_class_mean = np.zeros((X.shape[0], len(unique_classes)))
    row_means = np.mean(X, axis=1)
    for i in range(0, len(unique_classes)):
        ix = np.where(np.array(y) == unique_classes[i])[0]
        if ix.shape[0] > 1:
            row_class_mean[:, i] = np.mean(X[:, ix], axis=1).flatten()
        else:
            row_class_mean[:, i] = X[:, ix[0]].flatten()
    freq_counts = pd.DataFrame(y).value_counts()[unique_classes].values
    BBS = np.matmul(
        freq_counts,
        np.transpose(np.power(row_class_mean - np.atleast_2d(row_means), 2)),
    )  # (1,100)
    TSS = np.sum(np.power((X - np.atleast_2d(row_means)), 2), axis=1)  # (100,1)
    ESS = TSS.T - BBS
    df1 = len(unique_classes) - 1
    df2 = X.shape[1] - len(unique_classes)
    Fscore = (BBS / df1) / (ESS / df2)
    return np.asarray(Fscore)[0]  # (1,100)


def F_test_dense(X, y):
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


def F_test(X, y):
    if isinstance(X, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)):
        score = F_test_sparse(X, y)
    else:
        score = F_test_dense(X, y)
    return score


class Fselection:
    """Perform feature selection based on the F test. Be careful, the data needs to be normalized before feature selection based on the F-test can be applied.
    """

    def __init__(self, n_features=10):
        self.n_features = n_features

    def fit(
        self, data, labels
    ):  # don't need labfromels here but has to do with Pipeline construction
        data = np.log1p(data)
        F_scores = F_test(data.T, labels)
        if isinstance(F_scores, (pd.DataFrame, pd.Series)):
            idx = np.argsort(-(F_scores.values))
        else:
            idx = np.argsort(-(F_scores))
        self.selected_features = idx[0 : self.n_features]
        return self

    def transform(self, data):
        data = np.log1p(data)
        if isinstance(data, pd.DataFrame):
            return data.iloc[:, self.selected_features]
        else:
            return data[:, self.selected_features]

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


class HVGselection:
    """Perform highly variable feature selection based on the scanpy function.
    """
    def __init__(self, flavour, top_genes=500):
        self.flavour = flavour
        self.top_genes = top_genes

    def fit(
        self, data, labels
    ):  # don't need labels here but has to do with Pipeline construction
        adata = AnnData(data, dtype="float64")  # make sure adata is cont data (TRY)
        # sc.pp.normalize_total(adata) doesn't work with normalize after log
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(
            adata, n_top_genes=self.top_genes, flavor=self.flavour
        )
        self.highly_variable_genes = adata.var["highly_variable"]  # True/False list
        return self

    def transform(self, data):
        data = np.log1p(data)
        if isinstance(data, pd.DataFrame):
            return data.loc[:, self.highly_variable_genes]
        else:
            return data[:, self.highly_variable_genes]

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


########### ANALYSIS NO KFOLD ###########


#### Hierarchcial
def Run_H_NoKF(
    classifier, data, labels, parameters, n_jobsHCL, Norm=True, greedy_=False
):
    """Function to run hierarchical classification without K-fold cross validation with a dense data matrix.

    Parameters
    ----------
    classifier : 
        Scikit-learn classifier
    data : dense matrix
        Data matrix
    labels : list
        Cell type labels
    parameters : dict of str to sequence, or sequence of such
        Hyperparameters to be evaluated (for more information on the correct input format of this parameter check ParameterGrid by scikit-learn)
    n_jobsHCL : int
        Number of CPU cores used for parallelization of the hierarchical classification
    Norm : bool, optional
        Perform log(1+x) normalisation before running the analysis, by default True
    greedy_ : bool, optional
        Perform greedy (True) or non-greedy (False) hierarchical classification, by default False

    Returns
    -------
    Final Classifier:
        Trained scikit-lrean classifier
    Xtest: pandas dataframe
        Data test splits
    yest: list
        Label test splits
    predicted: list
        Predictions
    probs: matrix
        Prediction probabilities for all the classes
    Bestparam: float or int
        Best hyperparameter(s)

    """
    # Run without cross-validation to get one single metric at the end
    if Norm == True:
        data = np.log1p(data)  # log(1+x)

    Xtest, Xtrain, ytest, ytrain = train_test_split(
        data, labels, stratify=labels, train_size=0.25, random_state=6
    )

    Xtest_tuning, Xtrain_tuning, ytest_tuning, ytrain_tuning = train_test_split(
        Xtrain, ytrain, stratify=ytrain, train_size=0.25, random_state=6
    )

    i = 0
    param_grid = ParameterGrid(parameters)
    acc = np.zeros(len(list(param_grid)))
    for params in param_grid:
        clf = classifier.set_params(**params)
        clf_H = LCPN(clf, sep=";", n_jobs=n_jobsHCL, random_state=0, verbose=1)
        clf_H.fit(Xtrain_tuning, ytrain_tuning.values.ravel())
        y_pred, probs = clf_H.predict(
            Xtest_tuning, reject_thr=None, greedy=greedy_
        )  # is a list
        acc[i] = accuracy_score(
            ytest_tuning, y_pred
        )  # doesn't matter if ytest_tuning is an array or column vector
        i += 1

    acc = list(acc)
    max_accuracy = max(acc)
    index_bestparam = acc.index(max_accuracy)
    Bestparam = list(param_grid)[index_bestparam]

    Final_clf = classifier.set_params(**Bestparam)
    Final_Classifier = LCPN(
        Final_clf, sep=";", n_jobs=n_jobsHCL, random_state=0, verbose=1
    )
    Final_Classifier.fit(Xtrain, ytrain.values.ravel())
    predicted, probs = Final_Classifier.predict(Xtest, reject_thr=None, greedy=greedy_)
    accuracyfold = accuracy_score(ytest, predicted)

    print("accuracy fold", accuracyfold)

    return (Final_Classifier, Xtest, ytest, predicted, probs, Bestparam)


def Run_H_NoKF_sparse(
    classifier, data, labels, parameters, n_jobsHCL, Norm=True, greedy_=False
):
    """Function to run hierarchical classification without K-fold cross validation with a sparse data matrix.

    Parameters
    ----------
    classifier : 
        Scikit-learn classifier
    data : sparse matrix
        Data matrix
    labels : list
        Cell type labels
    parameters : dict of str to sequence, or sequence of such
        Hyperparameters to be evaluated (for more information on the correct input format of this parameter check ParameterGridd by scikit-learn)
    n_jobsHCL : int
        Number of CPU cores used for parallelization of the hierarchical classification
    Norm : bool, optional
        Perform log(1+x) normalisation before running the analysis, by default True
    greedy_ : bool, optional
        Perform greedy (True) or non-greedy (False) hierarchical classification, by default False

    Returns
    -------
    Final Classifier:
        Trained scikit-lrean classifier
    Xtest: pandas dataframe
        Data test splits
    yest: list
        Label test splits
    predicted: list
        Predictions
    probs: matrix
        Prediction probabilities for all the classes
    Bestparam: float or int
        Best hyperparameter(s)
    """
    # Run without cross-validation to get one single metric at the end
    if Norm == True:
        data = np.log1p(data)  # log(1+x)

    Xtest, Xtrain, ytest, ytrain = train_test_split(
        data, labels, stratify=labels, train_size=0.25, random_state=6
    )

    Xtest_tuning, Xtrain_tuning, ytest_tuning, ytrain_tuning = train_test_split(
        Xtrain, ytrain, stratify=ytrain, train_size=0.25, random_state=6
    )

    i = 0
    param_grid = ParameterGrid(parameters)
    acc = np.zeros(len(list(param_grid)))
    for params in param_grid:
        clf = classifier.set_params(**params)
        clf_H = LCPN(clf, sep=";", n_jobs=n_jobsHCL, random_state=0, verbose=1)
        clf_H.fit(Xtrain_tuning, ytrain_tuning.ravel())
        y_pred, probs = clf_H.predict(
            Xtest_tuning, reject_thr=None, greedy=greedy_
        )  # is a list
        acc[i] = accuracy_score(
            ytest_tuning, y_pred
        )  # doesn't matter if ytest_tuning is an array or column vector
        i += 1

    acc = list(acc)
    max_accuracy = max(acc)
    index_bestparam = acc.index(max_accuracy)
    Bestparam = list(param_grid)[index_bestparam]

    Final_clf = classifier.set_params(**Bestparam)
    Final_Classifier = LCPN(
        Final_clf, sep=";", n_jobs=n_jobsHCL, random_state=0, verbose=1
    )
    Final_Classifier.fit(Xtrain, ytrain.ravel())
    predicted, probs = Final_Classifier.predict(Xtest, reject_thr=None, greedy=greedy_)
    accuracyfold = accuracy_score(ytest, predicted)

    print("accuracy fold", accuracyfold)

    return (Final_Classifier, Xtest, ytest, predicted, probs, Bestparam)

#### Flat
def Run_Flat_NoKF(classifier, data, labels, parameters, Norm=True):
    """Function to run flat classification without K-fold cross validation.

    Parameters
    ----------
    classifier : 
        Scikit-learn classifier
    data : dense matrix
        Data matrix
    labels : list
        Cell type labels
    parameters : dict of str to sequence, or sequence of such
        Hyperparameters to be evaluated (for more information on the correct input format of this parameter check ParameterGridd by scikit-learn)
    Norm : bool, optional
        Perform log(1+x) normalisation before running the analysis, by default True

    Returns
    -------
    Final Classifier:
        Trained scikit-lrean classifier
    Xtest: pandas dataframe
        Data test splits
    yest: list
        Label test splits
    predicted: list
        Predictions
    """
    # Run without cross-validation to get one single metric at the end
    if Norm == True:
        data = np.log1p(data)  # log(1+x)

    Xtest, Xtrain, ytest, ytrain = train_test_split(
        data, labels, stratify=labels, train_size=0.25, random_state=6
    )

    Xtest_tuning, Xtrain_tuning, ytest_tuning, ytrain_tuning = train_test_split(
        Xtrain, ytrain, stratify=ytrain, train_size=0.25, random_state=6
    )

    i = 0
    param_grid = ParameterGrid(parameters)
    acc = np.zeros(len(list(param_grid)))
    for params in param_grid:
        print(params)
        clf = classifier.set_params(**params)
        clf.fit(Xtrain_tuning, ytrain_tuning.values.ravel())
        y_pred = clf.predict(Xtest_tuning)  # is a list
        acc[i] = accuracy_score(
            ytest_tuning, y_pred
        )  # doesn't matter if ytest_tuning is an array or column vector
        i += 1

    acc = list(acc)
    max_accuracy = max(acc)
    index_bestparam = acc.index(max_accuracy)
    Bestparam = list(param_grid)[index_bestparam]

    Final_Classifier = classifier.set_params(**Bestparam)
    Final_Classifier.fit(Xtrain, ytrain.values.ravel())

    predicted = Final_Classifier.predict(Xtest)
    accuracyfold = accuracy_score(ytest, predicted)

    print("accuracy fold", accuracyfold)
    return (Final_Classifier, Xtest, ytest, predicted)

########### ANALYSIS KFOLD ###########
#### Flat
def Run_Flat_KF_sparse(
    classifier_,
    n_folds,
    data,
    labels,
    parameters,
    Norm=True,
    HVG=False,
    F_test=False,
    save_clf=False,
    metric="accuracy_score",
):
    """Function to run flat classification with K-fold cross validation and a sparse data matrix.

    Parameters
    ----------
    classifier_ : 
        Scikit-learn classifier
    n_folds : int
        Number of folds, must be at least 2
    data : sparse matrix
        Data matrix
    labels : list
        Cell type labels
    parameters : dict of str to sequence, or sequence of such
        Hyperparameters to be evaluated (for more information on the correct input format of this parameter check ParameterGridd by scikit-learn)
    Norm : bool, optional
        Perform log(1+x) normalisation before running the analysis, by default True
    HVG : bool, optional
        If true, perform highly variable feature selection, the number of selected features ('top_genes') is a hyperparameter that should be incorporated in parameters, by default False
    F_test : bool, optional
        If True, perform feature selection based on the F-test, the number of selected features ('n_features') is a hyperparameter that should be incorporated in parameters, by default False
    save_clf : bool, optional
        If True, outputs the trained classifiers per fold together with the train-test splits, by default False
    metric : str, optional
        Use "accuracy score" or "log_loss" for test and train evaluation, by default "accuracy_score"

    Returns
    -------
    AllPredictedValues: list of lists
    AllProbabilities: list of matrices
    AllActualValues: list of lists
    AccuraciesFolds: list 
    Bestparams: list
    Classifiers: list of classifiers (optional)
    Xtests: list of matrices (optional)
    ytests: list of matrices (optional)
    """
    classifier_base = clone(classifier_)
    if Norm == True:
        data = np.log1p(data)  # log(1+x)

    if HVG == True:
        classifier = Pipeline(
            [
                ("feature_selection", HVGselection(flavour="seurat")),
                ("classification", classifier_base),
            ]
        )
    elif F_test == True:
        classifier = Pipeline(
            [("feature_selection", Fselection()), ("classification", classifier_base)]
        )
    else:
        classifier = classifier_base

    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1)

    Bestparams = []
    AccuraciesFolds = []
    AllPredictedValues = []
    AllProbabilities = []
    AllActualValues = []
    Classifiers = []
    Xtests = []
    ytests = []
    for train_idx, test_idx in folds.split(data, labels):
        X_test = data[test_idx, :]
        X_train = data[train_idx, :]
        y_test = labels[test_idx]
        y_train = labels[train_idx]

        Xtest_tuning, Xtrain_tuning, ytest_tuning, ytrain_tuning = train_test_split(
            X_train, y_train, stratify=y_train, train_size=0.25, random_state=6
        )
        i = 0
        param_grid = ParameterGrid(parameters)
        acc = np.zeros(len(list(param_grid)))
        for params in param_grid:
            print(params)
            classifier_tuning = classifier.set_params(**params)
            classifier_tuning.fit(Xtrain_tuning, ytrain_tuning.ravel())
            y_pred = classifier_tuning.predict(Xtest_tuning)  # is a list$
            # print(type(y_pred))
            if metric == "accuracy_score":
                acc[i] = accuracy_score(
                    ytest_tuning, y_pred
                )  # doesn't matter if ytest_tuning is an array or column vector
            elif metric == "log_loss":
                if not hasattr(classifier_base, "predict_proba"):
                    prob = predict_proba_from_scores(classifier_tuning, Xtest_tuning)
                else:
                    prob = classifier_tuning.predict_proba(Xtest_tuning)
                acc[i] = log_loss(ytest_tuning, prob)
            i += 1

        acc = list(acc)
        if metric == "accuracy_score":
            max_accuracy = max(acc)
        elif metric == "log_loss":
            max_accuracy = min(acc)
        index_bestparam = acc.index(max_accuracy)
        Bestparam = list(param_grid)[index_bestparam]
        Bestparams.append(Bestparam)

        Final_Classifier = classifier.set_params(**Bestparam)
        Final_Classifier.fit(X_train, y_train.ravel())
        if not hasattr(classifier_, "predict_proba"):
            prob = predict_proba_from_scores(Final_Classifier, X_test)
        else:
            prob = Final_Classifier.predict_proba(X_test)
        predicted = Final_Classifier.predict(X_test)
        
        if metric == "accuracy_score":
            accuracyfold = accuracy_score(y_test, predicted)
        elif metric == "log_loss":
            accuracyfold = log_loss(y_test, prob)
            
        AccuraciesFolds.append(accuracyfold)
        AllPredictedValues.append(list(predicted))
        AllProbabilities.append(prob)
        AllActualValues.append(list(y_test))
        Classifiers.append(Final_Classifier)
        Xtests.append(X_test)
        ytests.append(y_test)
        print("accuracy fold", accuracyfold)
    if save_clf:
        return (
            AllPredictedValues,
            AllProbabilities,
            AllActualValues,
            AccuraciesFolds,
            Bestparams,
            Classifiers,
            Xtests,
            ytests,
        )
    else:
        return (
            AllPredictedValues,
            AllActualValues,
            AccuraciesFolds,
            Bestparams,
        )



def Run_Flat_KF(
    classifier_,
    n_folds,
    data,
    labels,
    parameters,
    Norm=True,
    HVG=False,
    F_test=False,
    save_clf=False,
    metric="accuracy_score",
):
    """Function to run flat classification with K-fold cross validation and a dense data matrix.

    Parameters
    ----------
    classifier_ : 
        Scikit-learn classifier
    n_folds : int
        Number of folds, must be at least 2
    data : dense matrix
        Data matrix
    labels : list
        Cell type labels
    parameters : dict of str to sequence, or sequence of such
        Hyperparameters to be evaluated (for more information on the correct input format of this parameter check ParameterGridd by scikit-learn)
    Norm : bool, optional
        Perform log(1+x) normalisation before running the analysis, by default True
    HVG : bool, optional
        If true, perform highly variable feature selection, the number of selected features ('top_genes') is a hyperparameter that should be incorporated in parameters, by default False
    F_test : bool, optional
        If True, perform feature selection based on the F-test, the number of selected features ('n_features') is a hyperparameter that should be incorporated in parameters, by default False
    save_clf : bool, optional
        If True, outputs the trained classifiers per fold together with the data and label test splits, by default False
    metric : str, optional
        Use "accuracy score" or "log_loss" for test and train evaluation, by default "accuracy_score"

    Returns
    -------
    AllPredictedValues: list of lists
    AllProbabilities: list of matrices
    AllActualValues: list of lists
    AccuraciesFolds: list 
    Bestparams: list
    Classifiers: list of classifiers (optional)
    Xtests: list of matrices (optional)
    ytests: list of matrices (optional)
    """
    classifier_base = clone(classifier_)
    if Norm == True:
        data = np.log1p(data)  # log(1+x)

    if HVG == True:
        classifier = Pipeline(
            [
                ("feature_selection", HVGselection(flavour="seurat")),
                ("classification", classifier_base),
            ]
        )
    elif F_test == True:
        classifier = Pipeline(
            [("feaure_selection", Fselection()), ("classification", classifier_base)]
        )
    else:
        classifier = classifier_base

    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1)

    Bestparams = []
    AccuraciesFolds = []
    AllPredictedValues = []
    AllProbabilities = []
    AllActualValues = []
    Classifiers = []
    Xtests = []
    ytests = []
    for train_idx, test_idx in folds.split(data, labels):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        if not isinstance(labels, (pd.DataFrame, pd.Series)):
            labels = pd.DataFrame(labels)

        X_test = data.iloc[test_idx, :]
        X_train = data.iloc[train_idx, :]
        y_test = labels.iloc[test_idx]
        y_train = labels.iloc[train_idx]

        Xtest_tuning, Xtrain_tuning, ytest_tuning, ytrain_tuning = train_test_split(
            X_train, y_train, stratify=y_train, train_size=0.25, random_state=6
        )

        i = 0
        param_grid = ParameterGrid(parameters)
        acc = np.zeros(len(list(param_grid)))
        for params in param_grid:
            print(params)
            clf = classifier.set_params(**params)
            clf.fit(Xtrain_tuning, ytrain_tuning.values.ravel())
            y_pred = clf.predict(Xtest_tuning)  # is a list
            if metric == "accuracy_score":
                acc[i] = accuracy_score(ytest_tuning, y_pred)
            elif metric == "log_loss":
                if not hasattr(classifier_base, "predict_proba"):
                    prob = predict_proba_from_scores(clf, Xtest_tuning)
                else:
                    prob = clf.predict_proba(Xtest_tuning)
                acc[i] = log_loss(ytest_tuning, prob)
            i += 1

        acc = list(acc)
        if metric == "accuracy_score":
            max_accuracy = max(acc)
        elif metric == "log_loss":
            max_accuracy = min(acc)
        index_bestparam = acc.index(max_accuracy)
        Bestparam = list(param_grid)[index_bestparam]
        Bestparams.append(Bestparam)

        Final_Classifier = classifier.set_params(**Bestparam)
        Final_Classifier.fit(X_train, y_train.values.ravel())
        
        if not hasattr(classifier_base, "predict_proba"):
            prob = predict_proba_from_scores(Final_Classifier, X_test)
        else:
            prob = Final_Classifier.predict_proba(X_test)
            
        predicted = Final_Classifier.predict(X_test)

        if metric == "accuracy_score":
            accuracyfold = accuracy_score(y_test, predicted)
        elif metric == "log_loss":
            accuracyfold = log_loss(y_test, prob)
            
        AccuraciesFolds.append(accuracyfold)
        AllPredictedValues.append(list(predicted))
        AllProbabilities.append(prob)
        AllActualValues.append(list(y_test.values))
        Classifiers.append(Final_Classifier)
        Xtests.append(X_test)
        ytests.append(y_test)
        
        print("accuracy fold", accuracy_score(y_test, predicted))
    if save_clf:
        return (
            AllPredictedValues,
            AllProbabilities,
            AllActualValues,
            AccuraciesFolds,
            Bestparams,
            Classifiers,
            Xtests,
            ytests,
        )
    else:
        return (
            AllPredictedValues,
            AllActualValues,
            AccuraciesFolds,
            Bestparams,
        )

def Run_Flat_KF_sparse_splitted(
    # Runs seperately per fold
    classifier_,
    n_folds,
    data,
    labels,
    parameters,
    fold,
    Norm=True,
    HVG=False,
    F_test=False,
    save_clf=False,
    metric="accuracy_score",
):
    """Function to run flat classification with for one K-fold cross validation fold on a sparse data matrix.


    Parameters
    ----------
    classifier_ : 
        Scikit-learn classifier
    n_folds : int
        Number of folds, must be at least 2
    data : sparse matrix
        Data matrix
    labels : list
        Cell type labels
    parameters : dict of str to sequence, or sequence of such
        Hyperparameters to be evaluated (for more information on the correct input format of this parameter check ParameterGridd by scikit-learn)
    fold : int
        Specific fold that is currently considered
    Norm : bool, optional
        Perform log(1+x) normalisation before running the analysis, by default True
    HVG : bool, optional
        If true, perform highly variable feature selection, the number of selected features ('top_genes') is a hyperparameter that should be incorporated in parameters, by default False
    F_test : bool, optional
        If True, perform feature selection based on the F-test, the number of selected features ('n_features') is a hyperparameter that should be incorporated in parameters, by default False
    save_clf : bool, optional
        If True, outputs the trained classifiers per fold together with the data and labels test splits, by default False
    metric : str, optional
        Use "accuracy score" or "log_loss" for test and train evaluation, by default "accuracy_score"


    Returns
    -------
    predicted: list
    prob: matrix
    y_test; list
    accuracy_fold: float
    Bestparams: list
    acc: list
    Final_classifier: trained classifier (optional)

    """
    classifier_base = clone(classifier_)
    if Norm == True:
        data = np.log1p(data)  # log(1+x)

    if HVG == True:
        classifier = Pipeline(
            [
                ("feature_selection", HVGselection(flavour="seurat")),
                ("classification", classifier_base),
            ]
        )
    elif F_test == True:
        classifier = Pipeline(
            [("feature_selection", Fselection()), ("classification", classifier_base)]
        )
    else:
        classifier = classifier_base

    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1)

    Bestparams = []
    j = 1
    for train_idx, test_idx in folds.split(data, labels):
        if fold == j:
            X_test = data[test_idx, :]
            X_train = data[train_idx, :]
            y_test = labels[test_idx]
            y_train = labels[train_idx]

            Xtest_tuning, Xtrain_tuning, ytest_tuning, ytrain_tuning = train_test_split(
                X_train, y_train, stratify=y_train, train_size=0.25, random_state=6
            )
            i = 0
            param_grid = ParameterGrid(parameters)
            acc = np.zeros(len(list(param_grid)))
            for params in param_grid:
                print(params)
                classifier_tuning = classifier.set_params(**params)
                classifier_tuning.fit(Xtrain_tuning, ytrain_tuning.ravel())
                y_pred = classifier_tuning.predict(Xtest_tuning)  # is a list$
                # print(type(y_pred))
                if metric == "accuracy_score":
                    acc[i] = accuracy_score(
                        ytest_tuning, y_pred
                    )  # doesn't matter if ytest_tuning is an array or column vector
                elif metric == "log_loss":
                    if not hasattr(classifier_base, "predict_proba"):
                        prob = predict_proba_from_scores(classifier_tuning, Xtest_tuning)
                    else:
                        prob = classifier_tuning.predict_proba(Xtest_tuning)
                    acc[i] = log_loss(ytest_tuning, prob)
                i += 1

            acc = list(acc)
            if metric == "accuracy_score":
                max_accuracy = max(acc)
            elif metric == "log_loss":
                max_accuracy = min(acc)
            index_bestparam = acc.index(max_accuracy)
            Bestparam = list(param_grid)[index_bestparam]
            Bestparams.append(Bestparam)

            Final_Classifier = classifier.set_params(**Bestparam)
            Final_Classifier.fit(X_train, y_train.ravel())
            if not hasattr(classifier_, "predict_proba"):
                prob = predict_proba_from_scores(Final_Classifier, X_test)
            else:
                prob = Final_Classifier.predict_proba(X_test)
            predicted = Final_Classifier.predict(X_test)
            
            if metric == "accuracy_score":
                accuracyfold = accuracy_score(y_test, predicted)
            elif metric == "log_loss":
                accuracyfold = log_loss(y_test, prob)

            print("accuracy fold", accuracyfold)
            if save_clf:
                return (
                    list(predicted),
                    prob,
                    list(y_test),
                    accuracyfold,
                    Bestparams,
                    Final_Classifier,
                    X_test,
                    y_test,
                )
            
            else:
                return (
                    list(predicted),
                    list(y_test),
                    accuracyfold,
                    Bestparams,
                    acc,
                )
        else:
            j += 1
            continue



def Run_Flat_KF_splitted(
    classifier_,
    n_folds,
    data,
    labels,
    parameters,
    fold,
    Norm=True,
    HVG=False,
    F_test=False,
    save_clf=False,
    metric="accuracy_score",
):
    """Function to run flat classification with for one K-fold cross validation fold on a dense data matrix.

    Parameters
    ----------
    classifier_ : 
        Scikit-learn classifier
    n_folds : int
        Number of folds, must be at least 2
    data : dense matrix
        Data matrix
    labels : list
        Cell type labels
    parameters : dict of str to sequence, or sequence of such
        Hyperparameters to be evaluated (for more information on the correct input format of this parameter check ParameterGridd by scikit-learn)
    fold : int
        Specific fold that is currently considered
    Norm : bool, optional
        Perform log(1+x) normalisation before running the analysis, by default True
    HVG : bool, optional
        If true, perform highly variable feature selection, the number of selected features ('top_genes') is a hyperparameter that should be incorporated in parameters, by default False
    F_test : bool, optional
        If True, perform feature selection based on the F-test, the number of selected features ('n_features') is a hyperparameter that should be incorporated in parameters, by default False
    save_clf : bool, optional
        If True, outputs the trained classifiers per fold together with the data and label test splits, by default False
    metric : str, optional
        Use "accuracy score" or "log_loss" for test and train evaluation, by default "accuracy_score"


    Returns
    -------
    predicted: list
    prob: matrix
    X_test: matrix
    y_test: list
    accuracy_fold: float
    Bestparams: list
    acc: list
    Final_classifier: trained classifier (optional)
    """
    classifier_base = clone(classifier_)
    if Norm == True:
        data = np.log1p(data)  # log(1+x)

    if HVG == True:
        classifier = Pipeline(
            [
                ("feature_selection", HVGselection(flavour="seurat")),
                ("classification", classifier_base),
            ]
        )
    elif F_test == True:
        classifier = Pipeline(
            [("feaure_selection", Fselection()), ("classification", classifier_base)]
        )
    else:
        classifier = classifier_base

    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1)

    Bestparams = []
    j = 1
    for train_idx, test_idx in folds.split(data, labels):
        if fold == j:
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)

            if not isinstance(labels, (pd.DataFrame, pd.Series)):
                labels = pd.DataFrame(labels)

            X_test = data.iloc[test_idx, :]
            X_train = data.iloc[train_idx, :]
            y_test = labels.iloc[test_idx]
            y_train = labels.iloc[train_idx]

            Xtest_tuning, Xtrain_tuning, ytest_tuning, ytrain_tuning = train_test_split(
                X_train, y_train, stratify=y_train, train_size=0.25, random_state=6
            )

            i = 0
            param_grid = ParameterGrid(parameters)
            acc = np.zeros(len(list(param_grid)))
            for params in param_grid:
                clf = classifier.set_params(**params)
                clf.fit(Xtrain_tuning, ytrain_tuning.values.ravel())
                y_pred = clf.predict(Xtest_tuning)  # is a list
                if metric == "accuracy_score":
                    acc[i] = accuracy_score(ytest_tuning, y_pred)
                elif metric == "log_loss":
                    if not hasattr(classifier_base, "predict_proba"):
                        prob = predict_proba_from_scores(clf, Xtest_tuning)
                    else:
                        prob = clf.predict_proba(Xtest_tuning)
                    acc[i] = log_loss(ytest_tuning, prob)
                i += 1

            acc = list(acc)
            if metric == "accuracy_score":
                max_accuracy = max(acc)
            elif metric == "log_loss":
                max_accuracy = min(acc)
            index_bestparam = acc.index(max_accuracy)
            Bestparam = list(param_grid)[index_bestparam]
            Bestparams.append(Bestparam)

            Final_Classifier = classifier.set_params(**Bestparam)
            Final_Classifier.fit(X_train, y_train.values.ravel())
            
            if not hasattr(classifier_base, "predict_proba"):
                prob = predict_proba_from_scores(Final_Classifier, X_test)
            else:
                prob = Final_Classifier.predict_proba(X_test)
                
            predicted = Final_Classifier.predict(X_test)

            if metric == "accuracy_score":
                accuracyfold = accuracy_score(y_test, predicted)
            elif metric == "log_loss":
                accuracyfold = log_loss(y_test, prob)
            
            print("accuracy fold", accuracy_score(y_test, predicted))
            if save_clf:
                return (
                    list(predicted),
                    prob,
                    list(y_test.values),
                    accuracyfold,
                    Bestparams,
                    Final_Classifier,
                    X_test,
                    y_test,
                )
            else:
                return (
                    list(predicted),
                    list(y_test.values),
                    accuracyfold,
                    Bestparams,
                    acc,
                )
        else:
            j += 1
            continue
        

#### Hierarchical

def Run_H_KF(
    classifier_,
    n_folds,
    data,
    labels,
    parameters,
    n_jobsHCL,
    reject_thresh,
    greedy_=True,
    Norm=True,
    HVG=False,
    F_test=False,
    save_clf=False,
    metric="accuracy_score",
):
    """Function to run hierarchical classification with K-fold cross validation and a dense data matrix.

    Parameters
    ----------
    classifier_ : 
        Scikit-learn classifier
    n_folds : int
        Number of folds, must be at least 2
    data : dense matrix
        Data matrix
    labels : list
        Cell type labels
    parameters : dict of str to sequence, or sequence of such
        Hyperparameters to be evaluated (for more information on the correct input format of this parameter check ParameterGridd by scikit-learn)
    n_jobsHCL : int
        Number of CPU cores used for parallelization of the hierarchical classification
    reject_thresh : float or None (str)
        If not None, annotation will stop if the probability of a label will drop below the inputted valued
    greedy_ : bool, optional
        Perform greedy (True) or non-greedy (False) hierarchical classification, by default False
    Norm : bool, optional
        Perform log(1+x) normalisation before running the analysis, by default True
    HVG : bool, optional
        If true, perform highly variable feature selection, the number of selected features ('top_genes') is a hyperparameter that should be incorporated in parameters, by default False
    F_test : bool, optional
        If True, perform feature selection based on the F-test, the number of selected features ('n_features') is a hyperparameter that should be incorporated in parameters, by default False
    save_clf : bool, optional
        If True, outputs the trained classifiers per fold together with the data and label test splits, by default False
    metric : str, optional
        Use "accuracy score" or "log_loss" for test and train evaluation, by default "accuracy_score"
    Returns
    -------
    AllPredictedValues: list of lists
    AllProbabilities: list of matrices
    AllActualValues: list of lists
    AccuraciesFolds: list 
    Bestparams: list
    Classifiers: list of classifiers (optional)
    Xtests: list of matrices (optional)
    ytests: list of matrices (optional)
    """
    if Norm == True:
        data = np.log1p(data)  # log(1+x)

    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1)

    Bestparams = []
    AccuraciesFolds = []
    AllPredictedValues = []
    AllActualValues = []
    Classifiers = []
    Xtests = []
    ytests = []
    for train_idx, test_idx in folds.split(data, labels):
        print('new fold')
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        if not isinstance(labels, (pd.DataFrame, pd.Series)):
            labels = pd.DataFrame(labels)

        X_test = data.iloc[test_idx, :]
        X_train = data.iloc[train_idx, :]
        y_test = labels.iloc[test_idx]
        y_train = labels.iloc[train_idx]

        Xtest_tuning, Xtrain_tuning, ytest_tuning, ytrain_tuning = train_test_split(
            X_train, y_train, stratify=y_train, train_size=0.25, random_state=6
        )

        i = 0
        param_grid = ParameterGrid(parameters)
        acc = np.zeros(len(list(param_grid)))
        for params in param_grid:
            if HVG == True:
                params_classification = {
                    k.split("__")[1]: v
                    for k, v in params.items()
                    if k.startswith("classification__")
                }
                params_feature_selection = {
                    k.split("__")[1]: v
                    for k, v in params.items()
                    if k.startswith("feature_selection__")
                }
                classifier_ = classifier_.set_params(**params_classification)
                clf = LCPN(
                    Pipeline(
                        [
                            (
                                "feature_selection",
                                HVGselection(flavour="seurat").set_params(
                                    **params_feature_selection
                                ),
                            ),
                            ("classification", classifier_),
                        ]
                    ),
                    sep=";",
                    n_jobs=n_jobsHCL,
                    random_state=0,
                    verbose=1,
                )
            elif F_test == True:
                params_classification = {
                    k.split("__")[1]: v
                    for k, v in params.items()
                    if k.startswith("classification__")
                }
                params_feature_selection = {
                    k.split("__")[1]: v
                    for k, v in params.items()
                    if k.startswith("feature_selection__")
                }
                classifier_ = classifier_.set_params(**params_classification)
                clf = LCPN(
                    Pipeline(
                        [
                            (
                                "feature_selection",
                                Fselection().set_params(**params_feature_selection),
                            ),
                            ("classification", classifier_),
                        ]
                    ),
                    sep=";",
                    n_jobs=n_jobsHCL,
                    random_state=0,
                    verbose=1,
                )
            else:
                classifier_ = classifier_.set_params(**params)
                clf = LCPN(
                    classifier_, sep=";", n_jobs=n_jobsHCL, random_state=0, verbose=1
                )
            clf.fit(Xtrain_tuning, ytrain_tuning.values.ravel())
            y_pred, y_probs = clf.predict(
                Xtest_tuning, reject_thr=reject_thresh, greedy=greedy_
            )  # is a list$
            # print(type(y_pred))
            if metric == "accuracy_score":
                acc[i] = accuracy_score(ytest_tuning, y_pred)
            elif metric == "log_loss":
                acc[i] = log_loss(ytest_tuning, LCPN.predict_proba(clf, Xtest_tuning))

            i += 1

        acc = list(acc)
        print("training metrics in this fold", acc)
        if metric == "accuracy_score":
            max_accuracy = max(acc)
        elif metric == "log_loss":
            max_accuracy = min(acc)
        index_bestparam = acc.index(max_accuracy)
        Bestparam = list(param_grid)[index_bestparam]
        Bestparams.append(Bestparam)

        if HVG == True:
            Bestparams_classification = {
                k.split("__")[1]: v
                for k, v in Bestparam.items()
                if k.startswith("classification__")
            }
            Bestparams_feature_selection = {
                k.split("__")[1]: v
                for k, v in Bestparam.items()
                if k.startswith("feature_selection__")
            }
            classifier_ = classifier_.set_params(**Bestparams_classification)
            Final_Classifier = LCPN(
                Pipeline(
                    [
                        (
                            "feature_selection",
                            HVGselection(flavour="seurat").set_params(
                                **Bestparams_feature_selection
                            ),
                        ),
                        ("classification", classifier_),
                    ]
                ),
                sep=";",
                n_jobs=n_jobsHCL,
                random_state=0,
                verbose=1,
            )
        elif F_test == True:
            Bestparams_classification = {
                k.split("__")[1]: v
                for k, v in Bestparam.items()
                if k.startswith("classification__")
            }
            Bestparams_feature_selection = {
                k.split("__")[1]: v
                for k, v in Bestparam.items()
                if k.startswith("feature_selection__")
            }
            classifier_ = classifier_.set_params(**Bestparams_classification)
            Final_Classifier = LCPN(
                Pipeline(
                    [
                        (
                            "feature_selection",
                            Fselection().set_params(**Bestparams_feature_selection),
                        ),
                        ("classification", classifier_),
                    ]
                ),
                sep=";",
                n_jobs=n_jobsHCL,
                random_state=0,
                verbose=1,
            )
        else:
            classifier_ = classifier_.set_params(**params)
            Final_Classifier = LCPN(
                classifier_, sep=";", n_jobs=n_jobsHCL, random_state=0, verbose=1
            )
        Final_Classifier.fit(X_train, y_train.values.ravel())

        predicted, probs = Final_Classifier.predict(
            X_test, reject_thr=reject_thresh, greedy=greedy_
        )
        if metric == "accuracy_score":
            accuracyfold = accuracy_score(y_test, predicted)
        elif metric == "log_loss":
            accuracyfold = log_loss(
                y_test, LCPN.predict_proba(Final_Classifier, X_test)
            )
        AccuraciesFolds.append(accuracyfold)
        AllPredictedValues.append(list(predicted))
        AllActualValues.append(list(y_test.values))
        Classifiers.append(Final_Classifier)
        Xtests.append(X_test)
        ytests.append(y_test)
        print("test metric fold", accuracyfold)
        print("\n")
    if save_clf:
        return (
            AllPredictedValues,
            AllActualValues,
            AccuraciesFolds,
            Bestparams,
            Classifiers,
            Xtests,
            ytests,
        )
    else:
        return (
            AllPredictedValues,
            AllActualValues,
            AccuraciesFolds,
            Bestparams,

        )


def Run_H_KF_sparse(
    classifier_,
    n_folds,
    data,
    labels,
    parameters,
    n_jobsHCL,
    reject_thresh,
    greedy_=True,
    Norm=True,
    HVG=False,
    F_test=False,
    save_clf=False,
    metric="accuracy_score",
):
    """Function to run hierarchical classification with K-fold cross validation and a sparse data matrix.

    Parameters
    ----------
    classifier_ : 
        Scikit-learn classifier
    n_folds : int
        Number of folds, must be at least 2
    data : sparse matrix
        Data matrix
    labels : list
        Cell type labels
    parameters : dict of str to sequence, or sequence of such
        Hyperparameters to be evaluated (for more information on the correct input format of this parameter check ParameterGridd by scikit-learn)
    n_jobsHCL : int
        Number of CPU cores used for parallelization of the hierarchical classification
    reject_thresh : float or None (str)
        If not None, annotation will stop if the probability of a label will drop below the inputted value
    greedy_ : bool, optional
         Perform greedy (True) or non-greedy (False) hierarchical classification, by default False
    Norm : bool, optional
        Perform log(1+x) normalisation before running the analysis, by default True
    HVG : bool, optional
        If true, perform highly variable feature selection, the number of selected features ('top_genes') is a hyperparameter that should be incorporated in parameters, by default False
    F_test : bool, optional
        If True, perform feature selection based on the F-test, the number of selected features ('n_features') is a hyperparameter that should be incorporated in parameters, by default False
    save_clf : bool, optional
        If True, outputs the trained classifiers per fold together with the data and label test splits, by default False
    metric : str, optional
        Use "accuracy score" or "log_loss" for test and train evaluation, by default "accuracy_score"
    Returns
    -------
    AllPredictedValues: list of lists
    AllProbabilities: list of matrices
    AllActualValues: list of lists
    AccuraciesFolds: list 
    Bestparams: list
    Classifiers: list of classifiers (optional)
    Xtests: list of matrices (optional)
    ytests: list of matrices (optional)
    """
    if Norm == True:
        data = np.log1p(data)  # log(1+x)

    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1)

    Bestparams = []
    AccuraciesFolds = []
    AllAccuracies = []
    AllPredictedValues = []
    AllActualValues = []
    Classifiers = []
    Xtests = []
    ytests = []
    for train_idx, test_idx in folds.split(data, labels):
        X_test = data[test_idx, :]
        X_train = data[train_idx, :]
        y_test = labels[test_idx]
        y_train = labels[train_idx]

        Xtest_tuning, Xtrain_tuning, ytest_tuning, ytrain_tuning = train_test_split(
            X_train, y_train, stratify=y_train, train_size=0.25, random_state=6
        )
        i = 0
        param_grid = ParameterGrid(parameters)
        acc = np.zeros(len(list(param_grid)))
        for params in param_grid:
            if HVG == True:
                params_classification = {
                    k.split("__")[1]: v
                    for k, v in params.items()
                    if k.startswith("classification__")
                }
                params_feature_selection = {
                    k.split("__")[1]: v
                    for k, v in params.items()
                    if k.startswith("feature_selection__")
                }
                classifier_ = classifier_.set_params(**params_classification)
                clf = LCPN(
                    Pipeline(
                        [
                            (
                                "feature_selection",
                                HVGselection(flavour="seurat").set_params(
                                    **params_feature_selection
                                ),
                            ),
                            (
                                "classification",
                                classifier_,
                            ),
                        ]
                    ),
                    sep=";",
                    n_jobs=n_jobsHCL,
                    random_state=0,
                    verbose=1,
                )
            elif F_test == True:
                params_classification = {
                    k.split("__")[1]: v
                    for k, v in params.items()
                    if k.startswith("classification__")
                }
                params_feature_selection = {
                    k.split("__")[1]: v
                    for k, v in params.items()
                    if k.startswith("feature_selection__")
                }
                classifier_ = classifier_.set_params(**params_classification)
                clf = LCPN(
                    Pipeline(
                        [
                            (
                                "feature_selection",
                                Fselection().set_params(**params_feature_selection),
                            ),
                            (
                                "classification",
                                classifier_,
                            ),
                        ]
                    ),
                    sep=";",
                    n_jobs=n_jobsHCL,
                    random_state=0,
                    verbose=1,
                )
            else:
                classifier_ = classifier_.set_params(**params)
                clf = LCPN(
                    classifier_, sep=";", n_jobs=n_jobsHCL, random_state=0, verbose=1
                )
            clf.fit(Xtrain_tuning, ytrain_tuning.ravel())
            y_pred, y_probs = clf.predict(
                Xtest_tuning, reject_thr=reject_thresh, greedy=greedy_
            )  # is a list$
            # print(type(y_pred))
            if metric == "accuracy_score":
                acc[i] = accuracy_score(ytest_tuning, y_pred)
            elif metric == "log_loss":
                acc[i] = log_loss(
                    ytest_tuning, LCPN.predict_proba(clf, Xtest_tuning.todense())
                )
            i += 1

        acc = list(acc)
        print("training metrics in this fold", acc)
        if metric == "accuracy_score":
            max_accuracy = max(acc)
        elif metric == "log_loss":
            max_accuracy = min(acc)
        index_bestparam = acc.index(max_accuracy)
        Bestparam = list(param_grid)[index_bestparam]
        Bestparams.append(Bestparam)

        if HVG == True:
            Bestparams_classification = {
                k.split("__")[1]: v
                for k, v in Bestparam.items()
                if k.startswith("classification__")
            }
            Bestparams_feature_selection = {
                k.split("__")[1]: v
                for k, v in Bestparam.items()
                if k.startswith("feature_selection__")
            }
            classifier_ = classifier_.set_params(**Bestparams_classification)
            Final_Classifier = LCPN(
                Pipeline(
                    [
                        (
                            "feature_selection",
                            HVGselection(flavour="seurat").set_params(
                                **Bestparams_feature_selection
                            ),
                        ),
                        (
                            "classification",
                            classifier_,
                        ),
                    ]
                ),
                sep=";",
                n_jobs=n_jobsHCL,
                random_state=0,
                verbose=1,
            )
        elif F_test == True:
            Bestparams_classification = {
                k.split("__")[1]: v
                for k, v in Bestparam.items()
                if k.startswith("classification__")
            }
            Bestparams_feature_selection = {
                k.split("__")[1]: v
                for k, v in Bestparam.items()
                if k.startswith("feature_selection__")
            }
            classifier_ = classifier_.set_params(**Bestparams_classification)
            Final_Classifier = LCPN(
                Pipeline(
                    [
                        (
                            "feature_selection",
                            Fselection().set_params(**Bestparams_feature_selection),
                        ),
                        (
                            "classification",
                            classifier_,
                        ),
                    ]
                ),
                sep=";",
                n_jobs=n_jobsHCL,
                random_state=0,
                verbose=1,
            )
        else:
            classifier_ = classifier_.set_params(**params)
            Final_Classifier = LCPN(
                classifier_, sep=";", n_jobs=n_jobsHCL, random_state=0, verbose=1
            )
        Final_Classifier.fit(X_train, y_train.ravel())

        predicted, probs = Final_Classifier.predict(
            X_test, reject_thr=reject_thresh, greedy=greedy_
        )
        if metric == "accuracy_score":
            accuracyfold = accuracy_score(y_test, predicted)
        elif metric == "log_loss":
            accuracyfold = log_loss(
                y_test, LCPN.predict_proba(Final_Classifier, X_test.todense())
            )
        AccuraciesFolds.append(accuracyfold)
        AllPredictedValues.append(list(predicted))
        AllActualValues.append(list(y_test))
        Classifiers.append(Final_Classifier)
        Xtests.append(X_test)
        ytests.append(y_test)
        print("test metric fold", accuracyfold)
        print("\n")
    if save_clf:
        return (
            AllPredictedValues,
            AllActualValues,
            AccuraciesFolds,
            Bestparams,
            AllAccuracies,
            Classifiers,
            Xtests,
            ytests,
        )
    else:
        return (
            AllPredictedValues,
            AllActualValues,
            AccuraciesFolds,
            Bestparams,
            AllAccuracies,


        )
        
def Run_H_KF_splitted(
    classifier_,
    n_folds,
    data,
    labels,
    parameters,
    n_jobsHCL,
    reject_thresh,
    fold,
    greedy_=True,
    Norm=True,
    HVG=False,
    F_test=False,
    save_clf=False,
    metric="accuracy_score",
    ):
    """
    Function to run hierarchical classification with for one K-fold cross validation fold on a dense data matrix.

    Parameters
    ----------
    classifier_ : 
        Scikit-learn classifier
    n_folds : int
        Number of folds, must be at least 2
    data :  matrix
        Data matrix
    labels : list
        Cell type labels
    parameters : dict of str to sequence, or sequence of such
        Hyperparameters to be evaluated (for more information on the correct input format of this parameter check ParameterGridd by scikit-learn)
    fold : int
        Specific fold that is currently considered
    n_jobsHCL : int
        Number of CPU cores used for parallelization of the hierarchical classification
    reject_thresh : float or None (str)
        If not None, annotation will stop if the probability of a label will drop below the inputted value
    greedy_ : bool, optional
         Perform greedy (True) or non-greedy (False) hierarchical classification, by default False
    Norm : bool, optional
        Perform log(1+x) normalisation before running the analysis, by default True
    HVG : bool, optional
        If true, perform highly variable feature selection, the number of selected features ('top_genes') is a hyperparameter that should be incorporated in parameters, by default False
    F_test : bool, optional
        If True, perform feature selection based on the F-test, the number of selected features ('n_features') is a hyperparameter that should be incorporated in parameters, by default False
    save_clf : bool, optional
        If True, outputs the trained classifiers per fold together with the data and label test splits, by default False
    metric : str, optional
        Use "accuracy score" or "log_loss" for test and train evaluation, by default "accuracy_score"
    Returns
    -------
    predicted: list 
    y_test: list
    accuracy_fold: float
    Bestparam: float
    acc: list
    Final_Classifiers: classifiers (optional)
    Xtest: matrix (optional)
    """
    if Norm == True:
        data = np.log1p(data)  # log(1+x)

    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1)
    j = 1
    for train_idx, test_idx in folds.split(data, labels):
        print('new fold')
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        if not isinstance(labels, (pd.DataFrame, pd.Series)):
            labels = pd.DataFrame(labels)
        if j == fold:
            X_test = data.iloc[test_idx, :]
            X_train = data.iloc[train_idx, :]
            y_test = labels.iloc[test_idx]
            y_train = labels.iloc[train_idx]

            Xtest_tuning, Xtrain_tuning, ytest_tuning, ytrain_tuning = train_test_split(
                X_train, y_train, stratify=y_train, train_size=0.25, random_state=6
            )

            i = 0
            param_grid = ParameterGrid(parameters)
            acc = np.zeros(len(list(param_grid)))
            for params in param_grid:
                classifier_ = classifier_.set_params(**params)
                clf = LCPN(
                    classifier_, sep=";", n_jobs=n_jobsHCL, random_state=0, verbose=1
                )
                clf.fit(Xtrain_tuning, ytrain_tuning.values.ravel())
                y_pred, y_probs = clf.predict(
                    Xtest_tuning, reject_thr=reject_thresh, greedy=greedy_
                )  # is a list$
                # print(type(y_pred))
                if metric == "accuracy_score":
                    acc[i] = accuracy_score(ytest_tuning, y_pred)
                elif metric == "log_loss":
                    if not hasattr(classifier_, "predict_proba"):
                        prob = predict_proba_from_scores(clf, Xtest_tuning)
                    else:
                        prob = clf.predict_proba(Xtest_tuning)
                    acc[i] = log_loss(ytest_tuning, prob)

                i += 1

            acc = list(acc)
            print("training metrics in this fold", acc)
            if metric == "accuracy_score":
                max_accuracy = max(acc)
            elif metric == "log_loss":
                max_accuracy = min(acc)
            index_bestparam = acc.index(max_accuracy)
            Bestparam = list(param_grid)[index_bestparam]

            
            classifier_ = classifier_.set_params(**params)
            Final_Classifier = LCPN(
                classifier_, sep=";", n_jobs=n_jobsHCL, random_state=0, verbose=1
            )
            Final_Classifier.fit(X_train, y_train.values.ravel())

            predicted, probs = Final_Classifier.predict(
                X_test, reject_thr=reject_thresh, greedy=greedy_
            )
            if metric == "accuracy_score":
                accuracyfold = accuracy_score(y_test, predicted)
            elif metric == "log_loss":
                if not hasattr(classifier_, "predict_proba"):
                    prob = predict_proba_from_scores(Final_Classifier, X_test)
                else:
                    prob = Final_Classifier.predict_proba(X_test)
                accuracyfold = log_loss(
                    y_test,prob)
            print("test metric fold", accuracyfold)
            print("\n")
      
            if save_clf:
                return (
                    predicted,
                    y_test,
                    accuracyfold,
                    Bestparam,
                    acc,
                    Final_Classifier,
                    X_test,
                )
            else:
                return (
                    predicted,
                    y_test,
                    accuracyfold,
                    Bestparam,
                    acc,
                )
        else:
            j += 1
            continue
        
def Run_H_KF_sparse_splitted(
    classifier_,
    n_folds,
    data,
    labels,
    parameters,
    n_jobsHCL,
    reject_thresh,
    fold,
    greedy_=True,
    Norm=True,
    HVG=False,
    F_test=False,
    save_clf=False,
    metric="accuracy_score",
):
    """Function to run hierarchical classification with for one K-fold cross validation fold on a sparse data matrix.

    Parameters
    ----------
    classifier_ : 
        Scikit-learn classifier
    n_folds : int
        Number of folds, must be at least 2
    data : sparse matrix
        Data matrix
    labels : list
        Cell type labels
    parameters : dict of str to sequence, or sequence of such
        Hyperparameters to be evaluated (for more information on the correct input format of this parameter check ParameterGridd by scikit-learn)
    fold : int
        Specific fold that is currently considered
    n_jobsHCL : int
        Number of CPU cores used for parallelization of the hierarchical classification
    reject_thresh : float or None (str)
        If not None, annotation will stop if the probability of a label will drop below the inputted value
    greedy_ : bool, optional
         Perform greedy (True) or non-greedy (False) hierarchical classification, by default False
    Norm : bool, optional
        Perform log(1+x) normalisation before running the analysis, by default True
    HVG : bool, optional
        If true, perform highly variable feature selection, the number of selected features ('top_genes') is a hyperparameter that should be incorporated in parameters, by default False
    F_test : bool, optional
        If True, perform feature selection based on the F-test, the number of selected features ('n_features') is a hyperparameter that should be incorporated in parameters, by default False
    save_clf : bool, optional
        If True, outputs the trained classifiers per fold together with the data and label test splits, by default False
    metric : str, optional
        Use "accuracy score" or "log_loss" for test and train evaluation, by default "accuracy_score"
    Returns
    -------
    predicted: list 
    y_test: list
    accuracy_fold: float
    Bestparam: float
    acc: list
    Final_Classifiers: classifiers (optional)
    Xtest: matrix (optional)
    """
    if Norm == True:
        data = np.log1p(data)  # log(1+x)

    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1)
    j = 1
    for train_idx, test_idx in folds.split(data, labels):
        print('new fold')
        if j == fold:
            X_test = data[test_idx, :]
            X_train = data[train_idx, :]
            y_test = labels[test_idx]
            y_train = labels[train_idx]

            Xtest_tuning, Xtrain_tuning, ytest_tuning, ytrain_tuning = train_test_split(
                X_train, y_train, stratify=y_train, train_size=0.25, random_state=6
            )

            i = 0
            param_grid = ParameterGrid(parameters)
            acc = np.zeros(len(list(param_grid)))
            for params in param_grid:
                classifier_ = classifier_.set_params(**params)
                clf = LCPN(
                    classifier_, sep=";", n_jobs=n_jobsHCL, random_state=0, verbose=1
                )
                clf.fit(Xtrain_tuning, ytrain_tuning.ravel())
                y_pred, y_probs = clf.predict(
                    Xtest_tuning, reject_thr=reject_thresh, greedy=greedy_
                )  # is a list$
                # print(type(y_pred))
                if metric == "accuracy_score":
                    acc[i] = accuracy_score(ytest_tuning, y_pred)
                elif metric == "log_loss":
                    acc[i] = log_loss(ytest_tuning, LCPN.predict_proba(clf, Xtest_tuning.todense()))

                i += 1

            acc = list(acc)
            print("training metrics in this fold", acc)
            if metric == "accuracy_score":
                max_accuracy = max(acc)
            elif metric == "log_loss":
                max_accuracy = min(acc)
            index_bestparam = acc.index(max_accuracy)
            Bestparam = list(param_grid)[index_bestparam]

            
            classifier_ = classifier_.set_params(**params)
            Final_Classifier = LCPN(
                classifier_, sep=";", n_jobs=n_jobsHCL, random_state=0, verbose=1
            )
            Final_Classifier.fit(X_train, y_train.ravel())

            predicted, probs = Final_Classifier.predict(
                X_test, reject_thr=reject_thresh, greedy=greedy_
            )
            if metric == "accuracy_score":
                accuracyfold = accuracy_score(y_test, predicted)
            elif metric == "log_loss":
                accuracyfold = log_loss(
                    y_test, LCPN.predict_proba(Final_Classifier, X_test.todense())
                )
            print("test metric fold", accuracyfold)
            print("\n")
      
            if save_clf:
                return (
                    predicted,
                    y_test,
                    accuracyfold,
                    Bestparam,
                    acc,
                    Final_Classifier,
                    X_test,
                )
            else:
                return (
                    predicted,
                    y_test,
                    accuracyfold,
                    Bestparam,
                    acc,
                )
        else:
            j += 1
            continue

##### FUNCTION TO SAVE THE RESULTS
def SaveResultsKF(
    PredictedValues,
    ActualValues,
    AccuraciesFolds,
    AllAccuracies,
    overall_best_params,
    directory,
    namespecific,
):
    """Function to save K-Fold cross-validation results.

    Parameters
    ----------
    PredictedValues : list of lists
    ActualValues : list of lists
    AccuraciesFolds : list
    overall_best_params : list
    directory : str
        Directory where you want to save it
    namespecific : str
        Name of the analysis
        
    Return
    ------
    namespecific_Other.csv
        Contains accuracies and best parameters per fold
    namespecific_ActualValueslist.csv
    namespecific_Preslist.csv
    """
    os.chdir(directory)

    AllAccuracies.insert(0, "These are all the accuracies over the parameters per fold")
    AccuraciesFolds.insert(0, "The accuracy per folds")
    with open(namespecific + "_Other.csv", "w") as f1:
        write = csv.writer(f1)
        write.writerow(AccuraciesFolds)
        write.writerow(overall_best_params)
    f1.close()

    flatListAllActual = [item for elem in ActualValues for item in elem]
    with open(namespecific + "_ActualValueslist.csv", "w") as f2:
        writer = csv.writer(f2)
        writer.writerow(flatListAllActual)
    f2.close()

    flatListPred = [item for elem in PredictedValues for item in elem]
    with open(namespecific + "_Predslist.csv", "w") as f3:
        writer = csv.writer(f3)
        writer.writerow(flatListPred)
    f3.close()
