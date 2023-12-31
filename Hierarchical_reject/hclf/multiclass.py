"""
Code for hierarchical multi-class classifiers.
Authors: Thomas Mortier and Lauren Theunissen
Last update: 3/05/2023
"""
import csv
import time
import warnings

import numpy as np
import pandas as pd
from .utils import HLabelEncoder, PriorityQueue

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils import _message_with_time
from sklearn.utils.validation import check_X_y, check_array, check_random_state
from sklearn.exceptions import NotFittedError, FitFailedWarning
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed, parallel_backend
from collections import ChainMap

import matplotlib.pyplot as plt
import random


class LCPN(BaseEstimator, ClassifierMixin):
    """Local classifier per parent node (LCPN) classifier.

    Parameters
    ----------
    estimator : scikit-learn base estimator
        Represents the base estimator for the classification task in each node.
    sep : str, default=';'
        Path separator used for processing the hierarchical labels. If set to None,
        a random hierarchy is created and provided flat labels are converted,
        accordingly.
    k : tuple of int, default=(2,2)
        Min and max number of children a node can have in the random generated tree. Is ignored when
        sep is not set to None.
    n_jobs : int, default=None
        The number of jobs to run in parallel. Currently this applies to fit,
        and predict.
    random_state : RandomState or an int seed, default=None
        A random number generator instance to define the state of the
        random permutations generator.
    verbose : int, default=0
        Controls the verbosity: the higher, the more messages

    Examples
    --------
    >>> from hclf.multiclass import LCPN
    >>> from sklearn.linear_model import LogisticRegression
    >>>
    >>> clf = LCPN(LogisticRegression(random_state=0),
    >>>         sep=";",
    >>>         n_jobs=4,
    >>>         random_state=0,
    >>>         verbose=1)
    >>> clf.fit(X, y)
    >>> clf.score(X, y)
    """

    def __init__(
        self, estimator, sep=";", k=(2, 2), n_jobs=None, random_state=None, verbose=0
    ):
        self.estimator = clone(estimator)
        self.sep = sep
        self.k = k
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.tree = {}

    def _add_path(self, path):
        current_node = path[0]
        add_node = path[1]
        # check if add_node is already registred
        if add_node not in self.tree:
            # check if add_node is terminal
            if len(path) > 2:
                # register add_node to the tree
                self.tree[add_node] = {
                    "lbl": add_node,
                    "estimator": None,
                    "children": [],
                    "parent": current_node,
                }
            # add add_node to current_node's children (if not yet in list of children)
            if add_node not in self.tree[current_node]["children"]:
                self.tree[current_node]["children"].append(add_node)
            # set estimator when num. of children for current_node is higher than 1 and if not yet set
            if (
                len(self.tree[current_node]["children"]) > 1
                and self.tree[current_node]["estimator"] is None
            ):
                self.tree[current_node]["estimator"] = clone(self.estimator)
        else:
            # check for duplicate node labels
            if self.tree[add_node]["parent"] != current_node:
                warnings.warn(
                    "Duplicate node label {0} detected in hierarchy with parents {1}, {2}!".format(
                        add_node, self.tree[add_node]["parent"], current_node
                    ),
                    FitFailedWarning,
                )
        # process next couple of nodes in path
        if len(path) > 2:
            path = path[1:]
            self._add_path(path)

    def _fit_node(self, node):
        # check if node has estimator
        if node["estimator"] is not None:
            # transform data for node
            y_transform = []
            sel_ind = []
            for i, y in enumerate(self.y_):
                if node["lbl"] in y.split(self.sep):
                    # need to include current label and sample (as long as it's "complete")
                    y_split = y.split(self.sep)
                    if y_split.index(node["lbl"]) < len(y_split) - 1:
                        y_transform.append(y_split[y_split.index(node["lbl"]) + 1])
                        sel_ind.append(i)
            X_transform = self.X_[sel_ind, :]
            node["estimator"].fit(X_transform, y_transform)
            if self.verbose >= 2:
                print("Model {0} fitted!".format(node["lbl"]))
            # now make sure that the order of labels correspond to the order of children
            node["children"] = node["estimator"].classes_
        return {node["lbl"]: node}

    def fit(self, X, y):
        """Implementation of the fitting function for the LCPN classifier.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The class labels

        Returns
        -------
        self : object
            Returns self.
        """
        self.random_state_ = check_random_state(self.random_state)
        # need to make sure that X and y have the correct shape
        X, y = check_X_y(
            X, y, multi_output=False, accept_sparse=True
        )  # multi-output not supported (yet)
        # check if n_jobs is integer
        if not self.n_jobs is None:
            if not isinstance(self.n_jobs, int):
                raise TypeError("Parameter n_jobs must be of type int.")
        # store number of outputs and complete data seen during fit
        self.n_outputs_ = 1
        self.X_ = X
        self.y_ = y
        # store label of root node
        self.rlbl = self.y_[0].split(self.sep)[0]
        # init tree
        self.tree = {
            self.rlbl: {
                "lbl": self.rlbl,
                "estimator": None,
                "children": [],
                "parent": None,
            }
        }
        # check if sep is None or str
        if type(self.sep) != str and self.sep is not None:
            raise TypeError("Parameter sep must be of type str or None.")
        # init and fit the hierarchical model
        start_time = time.time()
        # first init the tree
        try:
            if self.sep is None:
                # transform labels to labels in some random hierarchy
                self.sep = ";"
                self.label_encoder_ = HLabelEncoder(
                    k=self.k, random_state=self.random_state_
                )
                self.y_ = self.label_encoder_.fit_transform(self.y_)
            else:
                self.label_encoder_ = None
            for lbl in self.y_:
                self._add_path(lbl.split(self.sep))
            # now proceed to fitting
            with parallel_backend("loky"):
                fitted_tree = Parallel(n_jobs=self.n_jobs)(
                    delayed(self._fit_node)(self.tree[node]) for node in self.tree
                )
            self.tree = {k: v for d in fitted_tree for k, v in d.items()}
        except NotFittedError as e:
            raise NotFittedError(
                "Tree fitting failed! Make sure that the provided data is in the correct format."
            )
        # now store classes (leaf nodes) seen during fit
        cls = []
        nodes_to_visit = [self.tree[self.rlbl]]
        while len(nodes_to_visit) > 0:
            curr_node = nodes_to_visit.pop()
            for c in curr_node["children"]:
                # check if child is leaf node
                if c not in self.tree:
                    cls.append(c)
                else:
                    # add child to nodes_to_visit
                    nodes_to_visit.append(self.tree[c])
        self.classes_ = cls
        # make sure that classes_ are in same format of original labels
        if self.label_encoder_ is not None:
            self.classes_ = self.label_encoder_.inverse_transform(self.classes_)
        else:
            # construct dict with leaf node lbls -> path mappings
            lbl_to_path = {yi.split(self.sep)[-1]: yi for yi in self.y_}
            self.classes_ = [lbl_to_path[cls] for cls in self.classes_]
        stop_time = time.time()
        if self.verbose >= 1:
            print(_message_with_time("LCPN", "fitting", stop_time - start_time))
        return self

    def _predict_greedy(self, i, X, scores, reject_thr, save_probs_path):
        # if reject_thr is None rejection will not be implemented
        preds = []
        probs = []
        # run over all samples
        for x in X:
            x = x.reshape(1, -1)
            pred = self.rlbl
            curr_node_prob = 1  # prob of root is 1
            pred_path = [pred]
            probs_path = [curr_node_prob]
            while pred in self.tree:
                curr_node = self.tree[pred]
                # check if we have a node with single path
                if curr_node["estimator"] is not None:
                    pred_probs = self._predict_proba(curr_node["estimator"], x, scores)
                    curr_node_prob_new = (
                        max(max(pred_probs)) * curr_node_prob
                    )  # gives an array apparently
                    pred = curr_node["estimator"].predict(x)[0]
                    if reject_thr != None and curr_node_prob_new < reject_thr:
                        break
                    else:
                        curr_node_prob = curr_node_prob_new
                        probs_path.append(curr_node_prob)
                else:
                    pred = curr_node["children"][0]
                    probs_path.append(1)
                pred_path.append(pred)
            preds.append(self.sep.join(pred_path))
            if save_probs_path:
                probs.append(self.sep.join(probs_path))
            else:
                probs.append(curr_node_prob)

        return {i: [preds, probs]}

    def _predict_ngreedy(self, i, X, scores, reject_thr, save_probs_path):
        # if reject_thr is None rejection will not be implemented
        preds = []
        probs = []
        # run over all samples
        for x in X:
            prob_path = []
            x = x.reshape(1, -1)
            nodes_to_visit = PriorityQueue()
            nodes_to_visit.push(1.0, self.rlbl)
            pred = None
            while not nodes_to_visit.is_empty():
                curr_node_prob, curr_node = nodes_to_visit.pop()
                curr_node_lbl = curr_node.split(self.sep)[-1]
                curr_node_prob = (
                    1 - curr_node_prob
                )  # has to do with heap implementation
                if reject_thr != None:
                    if curr_node_prob >= reject_thr:
                        optimal_node_prob = curr_node_prob
                        optimal_pred_path = curr_node
                        prob_path.append(curr_node_prob)
                    else:
                        break
                else:
                    prob_path.append(curr_node_prob)
                # check if we are at a leaf node
                if curr_node_lbl not in self.tree:
                    pred = curr_node
                    break
                else:
                    curr_node_v = self.tree[curr_node_lbl]
                    # check if we have a node with single path
                    if curr_node_v["estimator"] is not None:
                        # get probabilities
                        curr_node_ch_probs = self._predict_proba(
                            curr_node_v["estimator"], x, scores
                        )
                        # apply chain rule of probability
                        curr_node_ch_probs = curr_node_ch_probs * curr_node_prob
                        # add children to queue
                        for j, c in enumerate(curr_node_v["children"]):
                            prob_child = curr_node_ch_probs[:, j][0]
                            nodes_to_visit.push(prob_child, curr_node + self.sep + c)
                    else:
                        c = curr_node_v["children"][0]
                        nodes_to_visit.push(curr_node_prob, curr_node + self.sep + c)
                # save prob path here
            if reject_thr != None:
                if save_probs_path:
                    probs.append(optimal_node_prob)
                else:
                    probs.append(prob_path)
                preds.append(optimal_pred_path)
            else:
                if save_probs_path:
                    probs.append(prob_path)
                else:
                    probs.append(curr_node_prob)
                preds.append(pred)
        return {i: [preds, probs]}

    def predict(self, X, reject_thr=None, greedy=True, save_probs_path = False):
        """Return class predictions.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input samples.
        greedy: Boolean, default = True
             Returns Bayes-optimal solution when set to False. Returns
             solution by following the path of maximum probability in each node, otherwise.

        reject_thr : float, default=None
            If None, no rejection is implemented. if a float is given, classification occus until the level
            where probability prediction >= float.

        find_thr: Boolean, default = False
            If True, an accuracy - threshold graph will be constructed that will help you find an optimal rejection threshold

        Returns
        -------
        preds : ndarray
            Returns an array of predicted class labels.
        """
        # check input
        X = check_array(X, accept_sparse=True)
        scores = False
        preds = []
        probs = []
        start_time = time.time()
        # check whether the base estimator supports probabilities
        if not hasattr(self.estimator, "predict_proba"):
            # check whether the base estimator supports class scores
            if not hasattr(self.estimator, "decision_function"):
                raise NotFittedError(
                    "{0} does not support \
                         probabilistic predictions nor scores.".format(
                        self.estimator
                    )
                )
            else:
                scores = True
        try:
            # now proceed to predicting
            with parallel_backend("loky"):
                if greedy:
                    d = Parallel(n_jobs=self.n_jobs)(
                        delayed(self._predict_greedy)(i, X[ind], scores, reject_thr, save_probs_path)
                        for i, ind in enumerate(
                            np.array_split(range(X.shape[0]), self.n_jobs)
                        )
                    )
                else:
                    d = Parallel(n_jobs=self.n_jobs)(
                        delayed(self._predict_ngreedy)(i, X[ind], scores, reject_thr, save_probs_path)
                        for i, ind in enumerate(
                            np.array_split(range(X.shape[0]), self.n_jobs)
                        )
                    )
                # collect
                dictio = dict(ChainMap(*d))
            for k in np.sort(list(dictio.keys())):
                preds.extend(dictio[k][0])
                probs.extend(dictio[k][1])
            # in case of no predefined hierarchy, backtransform to original https://realpython.com/python-chainmap/#l labels
            if self.label_encoder_ is not None:
                preds = self.label_encoder_.inverse_transform(
                    [p.split(self.sep)[-1] for p in preds]
                )
        except NotFittedError as e:
            raise NotFittedError(
                "This model is not fitted yet. Cal 'fit' \
                    with appropriate arguments before using this \
                    method."
            )
        stop_time = time.time()
        if self.verbose >= 1:
            print(_message_with_time("LCPN", "predicting", stop_time - start_time))
        return (preds, probs)

    def _predict_proba(self, estimator, X, scores=False):
        if not scores:
            return estimator.predict_proba(X)
        else:
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
        
    def predict_proba_full(self, X, pipeline=False):
        """We traverse over the fitted tree and save the output probabilities per classifier in a dictionary, with propagation of the probabilities"""
        # check input
        X = check_array(X)
        scores = False
        results_dict = {}
        start_time = time.time()
        if not hasattr(self.estimator, "predict_proba"):
            # check whether the base estimator supports class scores
            if not hasattr(self.estimator, "decision_function"):
                raise NotFittedError(
                    "{0} does not support \
                        probabilistic predictions nor scores.".format(
                        self.estimator
                    )
                )
            else:
                scores = True

        try:
            nodes_to_visit = [(self.tree[self.rlbl], np.ones((X.shape[0], 1)))]
            while len(nodes_to_visit) > 0:
                curr_node, parent_prob = nodes_to_visit.pop()
                # check if we have a node with single path
                if curr_node["estimator"] is not None:
                    # get probabilities
                    curr_node_probs = self._predict_proba(
                        curr_node["estimator"], X, scores
                    )
                    # apply chain rule of probability
                    curr_node_probs = curr_node_probs * parent_prob

                    # Store the results for the estimator node
                    results_dict[curr_node["lbl"]] = {
                        "probs": curr_node_probs,
                        "classes": curr_node["estimator"].classes_,
                    }
                    for i, c in enumerate(curr_node["children"]):
                        # check if child is leaf node
                        prob_child = curr_node_probs[:, i].reshape(-1, 1)
                        if c in self.tree:
                            nodes_to_visit.append((self.tree[c], prob_child))
                else:
                    results_dict[curr_node["lbl"]] = {
                        "probs": parent_prob,
                        "classes": np.array([curr_node["children"][0]]),
                    }
                    c = curr_node["children"][0]
                    # check if child is leaf node
                    if c in self.tree:
                        nodes_to_visit.append((self.tree[c], parent_prob))
        except NotFittedError as e:
            raise NotFittedError(
                "This model is not fitted yet. Cal 'fit' \
                    with appropriate arguments before using this \
                    method."
            )
        return results_dict
    
    def predict_proba(self, X, pipeline=False):
        """Return probability estimates.

        Important: the returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input samples.
        avg : boolean, default=True
            Return model average when true, and array of probability estimates otherwise.

        Returns
        -------
        probs : ndarray
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in self.classes_.
        """
        # check input
        X = check_array(X)
        scores = False
        probs = []
        start_time = time.time()
        if not hasattr(self.estimator, "predict_proba"):
            # check whether the base estimator supports class scores
            if not hasattr(self.estimator, "decision_function"):
                raise NotFittedError(
                    "{0} does not support \
                        probabilistic predictions nor scores.".format(
                        self.estimator
                    )
                )
            else:
                scores = True

        try:
            nodes_to_visit = [(self.tree[self.rlbl], np.ones((X.shape[0], 1)))]
            while len(nodes_to_visit) > 0:
                curr_node, parent_prob = nodes_to_visit.pop()
                # check if we have a node with single path
                if curr_node["estimator"] is not None:
                    # get probabilities
                    curr_node_probs = self._predict_proba(
                        curr_node["estimator"], X, scores
                    )
                    # apply chain rule of probability
                    curr_node_probs = curr_node_probs * parent_prob
                    for i, c in enumerate(curr_node["children"]):
                        # check if child is leaf node
                        prob_child = curr_node_probs[:, i].reshape(-1, 1)
                        if c not in self.tree:
                            probs.append(prob_child)
                        else:
                            # add child to nodes_to_visit
                            nodes_to_visit.append((self.tree[c], prob_child))
                else:
                    c = curr_node["children"][0]
                    # check if child is leaf node
                    if c not in self.tree:
                        probs.append(parent_prob)
                    else:
                        # add child to nodes_to_visit
                        nodes_to_visit.append((self.tree[c], parent_prob))
        except NotFittedError as e:
            raise NotFittedError(
                "This model is not fitted yet. Cal 'fit' \
                    with appropriate arguments before using this \
                    method."
            )
        stop_time = time.time()
        if self.verbose >= 1:
            print(
                _message_with_time(
                    "LCPN", "predicting probabilities", stop_time - start_time
                )
            )
        return np.hstack(probs)

    def score(self, X, y):
        """Return mean accuracy score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        # check input and outputs
        X, y = check_X_y(X, y, multi_output=False)
        start_time = time.time()
        try:
            preds = self.predict(X)
        except NotFittedError as e:
            raise NotFittedError(
                "This model is not fitted yet. Cal 'fit' \
                    with appropriate arguments before using this \
                    method."
            )
        stop_time = time.time()
        if self.verbose >= 1:
            print(
                _message_with_time("LCPN", "calculating score", stop_time - start_time)
            )
        score = accuracy_score(y, preds)
        return score

    def score_nodes(self, X, y):
        """Return mean accuracy score for each node in the hierarchy.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.

        Returns
        -------
        score_dict : dict
            Mean accuracy of self.predict(X) wrt. y for each node in the hierarchy.
        """
        # check input and outputs
        X, y = check_X_y(X, y, multi_output=False)
        start_time = time.time()
        score_dict = {}
        try:
            # transform the flat labels, in case of no predefined hierarchy
            if self.label_encoder_ is not None:
                y = self.label_encoder_.transform(y)
            for node in self.tree:
                node = self.tree[node]
                # check if node has estimator
                if node["estimator"] is not None:
                    # transform data for node
                    y_transform = []
                    sel_ind = []
                    for i, yi in enumerate(y):
                        if node["lbl"] in yi.split(self.sep):
                            # need to include current label and sample (as long as it's "complete")
                            y_split = yi.split(self.sep)
                            if y_split.index(node["lbl"]) < len(y_split) - 1:
                                y_transform.append(
                                    y_split[y_split.index(node["lbl"]) + 1]
                                )
                                sel_ind.append(i)
                    X_transform = X[sel_ind, :]
                    if len(sel_ind) != 0:
                        # obtain predictions
                        node_preds = node["estimator"].predict(X_transform)
                        acc = accuracy_score(y_transform, node_preds)
                        score_dict[node["lbl"]] = acc
        except NotFittedError as e:
            raise NotFittedError(
                "This model is not fitted yet. Cal 'fit' \
                    with appropriate arguments before using this \
                    method."
            )
        stop_time = time.time()
        if self.verbose >= 1:
            print(
                _message_with_time(
                    "LCPN", "calculating node scores", stop_time - start_time
                )
            )
        return score_dict

    def accuracy_score_reject(self, ytrue, ypred):
        """_summary_

        Args:
            ytrue list of shape (n_samples,) or (n_samples, n_outputs):
                The tue labels
            ypred list of shape (n_samples,) or (n_samples, n_outputs):
                The predictions for the true labels
        """
        if isinstance(ytrue, pd.Series):
            ytrue = ytrue.values.tolist()
        if isinstance(ytrue, pd.DataFrame):
            ytrue = ytrue.iloc[:, 0].values.tolist()
        if isinstance(ytrue, np.ndarray):
            ytrue = ytrue.ravel().tolist()

        if isinstance(ypred, pd.Series):
            ypred = ypred.values.tolist()
        if isinstance(ypred, np.ndarray):
            ypred = ypred.tolist()

        ytrue_adjusted = [
            (";").join(ytrue[i].split(";")[0 : len(ypred[i].split(";"))])
            if len(ypred[i].split(";")) < len(ytrue[i].split(";"))
            else ytrue[i]
            for i in range(0, len(ytrue))
        ]
        return accuracy_score(ytrue_adjusted, ypred)

    def find_percentage_reject(self, ytrue, ypred):
        count = 0
        for i in range(0, len(ytrue)):
            len_pred = len(ypred[i].split(";"))
            len_ytrue = len(ytrue[i].split(";"))
            if len_ytrue > len_pred:
                count += 1

        return (count / len(ytrue)) * 100

    def Accuracy_Rejection_noReject(
        self, X, y, fig_title, step=0.005, save_results=None, save_fig_title=None
    ):
        """_summary_

        Plots and or saves accuracy scores based on rejection thresholds.
        Here rejection is not explicitely performed. Rejected labels are just not included in the analysis

        'Note you need a fitted classifier before this'
        Parameters
        ----------
        X : test data
        y : test labels
        fig_title : Title of the plot
        step : float, optional
            steps in between rejection thresholds, by default 0.01
        save_results : _type_, optional
            by default None
        save_fig_title : _type_, optional
            by default None
        """
        steps = np.arange(0, 1 + step, step)
        acc = np.zeros(len(steps))
        predicted, probs = self.predict(X)
        for i in range(0, len(steps)):
            t = steps[i]
            idx_keep = [i > t for i in probs]
            if sum(idx_keep) == 0:
                acc[i] = 1
            else:
                acc[i] = accuracy_score(
                    np.array(y)[idx_keep], np.array(predicted)[idx_keep]
                )
        plt.scatter(steps, acc)
        plt.ylim(0, 1)
        plt.xlabel("Rejection Threshold")
        plt.ylabel("Accuracy score")
        plt.title(fig_title)
        if save_fig_title is not None:
            plt.savefig(save_fig_title, bbox_inches="tight")

        if save_results is not None:
            with open(save_results, "w") as f:
                writer = csv.writer(f)
                writer.writerow(acc)
                writer.writerow(steps)

    def Accuracy_Rejection(
        self,
        X,
        y,
        fig_title,
        step=0.01,
        greedy_=True,
        save_results=None,
        save_fig_title=None,
    ):
        """_summary_

        Plots and or saves accuracy scores based on rejection thresholds.
        Here rejection is not explicitely performed. Rejected labels are just not included in the analysis

        Parameters
        ----------
        X : test data
        y : test labels
        fig_title : Title of the plot
        greedy_: Boolean, optional
        step : float, optional
            steps in between rejection thresholds, by default 0.01
        save_results : _type_, optional
            by default None
        save_fig_title : _type_, optional
            by default None
        """
        step = 0.01
        steps = np.arange(0, 1 + step, step)
        acc = np.zeros(len(steps))

        for i in range(0, len(steps)):
            t = steps[i]
            predicted, probs = self.predict(X, reject_thr=t, greedy=greedy_)
            acc[i] = self.accuracy_score_reject(y, predicted)

        plt.scatter(steps, acc)
        plt.ylim(0, 1)
        plt.xlabel("Rejection Threshold")
        plt.ylabel("Accuracy score")
        plt.title(fig_title)
        if save_fig_title is not None:
            plt.savefig(save_fig_title, bbox_inches="tight")

        if save_results is not None:
            with open(save_results, "w") as f:
                writer = csv.writer(f)
                writer.writerow(acc)
                writer.writerow(steps)

