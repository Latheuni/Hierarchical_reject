import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


# Utility functions


def check_list_of_lists(li_1):
    """Check for presence list of lists"""
    return any(isinstance(i, list) for i in li_1)


def flatten_list_of_lists(l):
    """Flattens list of lists to a normal list"""
    return [item for sublist in l for item in sublist]
    
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

def find_percentage_reject(ytrue, ypred):
    """Returns the percentage of labels that are rejected

    Parameters
    ----------
    ytrue : list
        True labels
    ypred : list
        Predicted labels

    Returns
    -------
    Float
        Percentage of rejected labels
    """
    # Assert correct begining datatype
    assert isinstance(
        ytrue, list
    ), "find_precentage_reject -- ytrue needs to be an list"
    if check_list_of_lists(ytrue):
        print("find_precentage_reject -- Carefull, ytrue was a list of lists")
        ytrue = flatten_list_of_lists(ytrue)

    assert isinstance(
        ypred, list
    ), "find_precentage_reject -- ypred needs to be an list"
    if check_list_of_lists(ypred):
        print("find_precentage_reject -- Carefull, ypred was a list of lists")
        ypred = flatten_list_of_lists(ypred)

    count = 0
    lengths_ytrue = []
    lengths_pred = []
    for i in range(0, len(ytrue)):
        len_pred = len(ypred[i].split(";"))
        len_ytrue = len(ytrue[i].split(";"))
        if len_ytrue > len_pred:
            count += 1
        lengths_ytrue.append(len_ytrue)
        lengths_pred.append(len_pred)

    return count / len(ytrue), lengths_pred, lengths_ytrue


def _find_indices(list_to_check, item_to_find):
    """Returns indices of all occurences of item_to_find in list_to_check"""
    return [idx for idx, value in enumerate(list_to_check) if value == item_to_find]


def _children_to_parent(labels):
    """Makes a simple conversion dictionary: child --> parent"""
    unique_labels = np.unique(labels)
    conversion_dict = dict()
    for lbl in unique_labels:
        max_len_lbl = len(lbl.split(";"))
        for index in reversed(range(1, max_len_lbl)):
            if lbl.split(";")[index] not in conversion_dict.keys():
                child = lbl.split(";")[index]
                conversion_dict[child] = lbl.split(";")[index - 1]
    return conversion_dict


def accuracy_score_reject_intermediate(ytrue, ypred):
    """Calculates the accuracy score, intermediate rejected labels that are correct are considered as TP

    Args:
        ytrue list:
            The tue labels
        ypred list:
            The predictions for the true labels
    """
    # Assert correct beginng datatype
    assert isinstance(ytrue, list), "accuracy_score_reject -- ytrue needs to be an list"
    assert isinstance(ypred, list), "accuracy_score_reject - ypred needs to be an list"

    # Check for presence of list of lists
    if check_list_of_lists(ytrue):
        print("accuracy_score_reject -- Carefull, ytrue was a list of lists")
        ytrue = flatten_list_of_lists(ytrue)

    if check_list_of_lists(ypred):
        print("accuracy_score_reject -- Carefull, ypred was a list of lists")
        ypred = flatten_list_of_lists(ypred)

    # Adjust True label to correct length for comparison if predicted label is rejected
    ytrue_adjusted = [
        (";").join(ytrue[i].split(";")[0 : len(ypred[i].split(";"))])
        if len(ypred[i].split(";")) < len(ytrue[i].split(";"))
        else ytrue[i]
        for i in range(0, len(ytrue))
    ]
    return accuracy_score(ytrue_adjusted, ypred)

def accuracy_score_reject_delete(ytrue, ypred):
    """Calculates the accuracy score, rejected labels are not considered

    Args:
        ytrue list:
            The tue labels
        ypred list:
            The predictions for the true labels
    """
    # Assert correct beginng datatype
    assert isinstance(ytrue, list), "accuracy_score_reject -- ytrue needs to be an list"
    assert isinstance(ypred, list), "accuracy_score_reject - ypred needs to be an list"

    # Check for presence of list of lists
    if check_list_of_lists(ytrue):
        print("accuracy_score_reject -- Carefull, ytrue was a list of lists")
        ytrue = flatten_list_of_lists(ytrue)

    if check_list_of_lists(ypred):
        print("accuracy_score_reject -- Carefull, ypred was a list of lists")
        ypred = flatten_list_of_lists(ypred)


    rejected = [True if label in np.unique(ytrue) else False for label in ypred]
    
    filtered_ytrue = [ytrue[i] for i in range(len(ytrue)) if rejected[i]]
    filtered_ypred= [ypred[i] for i in range(len(ytrue)) if rejected[i]]
    
    if filtered_ytrue == []:
        return(1.0)
    else:
        return accuracy_score(filtered_ytrue, filtered_ypred)


def _tree_flat(Labels, classes, balanced):  # WORKS
    """Records the Tree hierarchy present in the True labels

    Parameters
    ----------
    Labels : list
        true labels
    classes : list
        classes of the Labels
    balanced : Boolean
        inidcates of the heirarchy in Labels is balanced

    Returns
    -------
    dict
        dictionary with fior every node label it's parents
    """
    "Records the Tree hierarchy present in the True labels"
    d = dict()

    # Reformat unbalanced labels and split labels into sublabels
    if balanced:
        Labels_new, classes = _handle_unbalanced_labels(
            Labels, balanced=True, flat=True
        )
    else:
        Labels_new, classes, conversion_extra_labels = _handle_unbalanced_labels(
            Labels, classes, balanced=False, flat=True
        )
    labels_splitted = [i.split(";") for i in Labels_new]

    # Loop over the label levels and construct conversion dict
    for level in range(1, max([len(j) for j in labels_splitted])):
        unique_labels_level = [m[level] for m in labels_splitted]
        for ulbl in unique_labels_level:
            if level == 0:
                d[ulbl] = None
            else:
                first_occ = unique_labels_level.index(ulbl)
                parents = labels_splitted[first_occ][level - 1]
                d[ulbl] = parents
    return d


def _handle_unbalanced_labels(labels, classes, balanced=False):
    """Function that will check the presence of imbalance and add extra labels to unbalanced labels
        if necessary to create a balanced tree

    Parameters
    ----------
    labels : list
        true labels
    classes : list
        classes present in Labels
    balanced : bool, optional
       inidcates the presence of balanced labels, by default False

    Returns
    -------
    list, list, dict optional
        adapted classes and labels and conversion_dict if unbalanced (key original label, item adapted label)
    """

    # Construct child --> parent dictionary for later
    d_child_parent = _children_to_parent(labels)

    # Calculate maximum amount of levels
    max_len = max([len(l.split(";")) for l in labels])

    # Extract unbalanced labels
    x = [len(l.split(";")) < max_len for l in labels]
    unbalanced_labels = np.unique(np.array(labels)[x])

    # Generate index indicating unbalanced labels or not
    conversion_unbalanced = dict()
    rev_conversion_unbalanced = dict()
    conversion_extra_labels = dict()

    if balanced:
        # No conversion needed
        return labels, classes

    else:
        # Construct conversion_unbalanced dict, that transforms unbalanced labels into full lentgh labels
        for unbalanced_l in unbalanced_labels:
            # Register these for later
            original = unbalanced_l
            last_sublabel = original.split(";")[-1]

            count = 1

            while len(unbalanced_l.split(";")) < max_len:  # Starts from leaf node
                z = unbalanced_l.split(";")
                z.append("EXTRA_" + unbalanced_l.split(";")[-1])
                unbalanced_l = (";").join(z)

                # Construct conversion_extra_labels so that meaningful classifiers can be found back
                if unbalanced_l.split(";")[-1] not in conversion_extra_labels.keys():
                    conversion_extra_labels[
                        unbalanced_l.split(";")[-1]
                    ] = d_child_parent[last_sublabel]

                ## Add leaf nodes to the conversion_extra_labels, as leaf nodes don't have estimators (prob from parent node)
                if last_sublabel not in conversion_extra_labels.keys():
                    conversion_extra_labels[last_sublabel] = d_child_parent[
                        last_sublabel
                    ]

                count += 1
            conversion_unbalanced[original] = unbalanced_l
            rev_conversion_unbalanced[unbalanced_l] = original

        return (
            [conversion_unbalanced[l] if l in unbalanced_labels else l for l in labels],
            [
                conversion_unbalanced[l] if l in unbalanced_labels else l
                for l in classes
            ],
            rev_conversion_unbalanced,
        )


# Flat mimic hierarchical functions
def full_prob_levels_from_flat(classes, uni_classes, probability_matrix):
    """Construct Level probability matrix for parent classes

    Parameters
    ----------
    classes: list
        parent classes for classes one level lower
    uni_classest : np.array
        unique list from classes (mainly important order)
    probability_matrix : np.matrix
        probability matrix for the children labels

    Returns
    -------
    _type_
        _description_
    """
    count = 0

    # For every unique class
    for parent in uni_classes:
        # find out which corresponding children have that class
        ind = _find_indices(classes, parent)

        # Calculate for that parent class the probability matrix
        if count == 0:
            prob_ = probability_matrix[:, ind].sum(axis=1)
            prob = prob_.reshape(prob_.shape[0], 1)
        else:
            x_ = probability_matrix[:, ind].sum(axis=1)
            x = x_.reshape(x_.shape[0], 1)
            prob = np.concatenate((prob, x), axis=1)

        count += 1
    return prob


def hierprob_from_flat_non_greedy(
    probability_matrix, classes, predictions, ytrue, balanced=False
):
    """_summary_

    _extended_summary_

    Parameters
    ----------
    probability_matrix : matrix
        flat probability matrix outputted by the classifier
    classes : list
        unique classes seen by the classifier
    predictions : list
        predicted class labels
    ytrue : list
        _true class labels
    balanced : bool, optional
        inidcates the presence of balanced labels by default False

    Returns
    -------
    pd.DataFrame,
        probability df with recorded proabbilities for every sublabel

    list,
        converted true labels (unbalanced --> balanced)

    list,
        converted pred labels (unbalanced --> balanced)
    """

    labels_splitted = [i.split(";") for i in predictions]
    max_levels = max([len(j) for j in labels_splitted])

    # Convert labels with extra labels
    if balanced:
        Labels_c_pred, classes_c_pred = _handle_unbalanced_labels(
            predictions,
            classes,
            balanced=True,  # Possibility that predictions don't have labels present in the classes
        )
        Labels_c_true, classes_c_true = _handle_unbalanced_labels(
            ytrue, classes, balanced=True
        )
    else:
        Labels_c_pred, classes_c_pred, _ = _handle_unbalanced_labels(
            predictions,
            classes,
            balanced=False,
        )
        Labels_c_true, classes_c_true, conversion_d = _handle_unbalanced_labels(
            ytrue, classes, balanced=False
        )

    for s in reversed(range(1, max_levels)):
        prob_ = probability_matrix
        if s == max_levels - 1:
            # Store most probable label lowest level (in dict with label)
            final_prob = probability_matrix.max(
                axis=1
            )  # Take max element per row, this is the final label (non-greedy)
            df = pd.DataFrame(final_prob)
            df.index = ytrue

        else:
            # labels final level
            curr_labels = [i.split(";")[s] for i in Labels_c_pred]
            classes_last = [
                i.split(";")[s] for i in classes_c_true
            ]  # Incorporate all classes, not just those that were predicted
            unique_classes_last = np.unique(classes_last)

            # Get indices predicted class labels in probability matrix
            d = {item: idx for idx, item in enumerate(unique_classes_last)}
            idx_curr_labels = [d[i] for i in curr_labels]

            # Calculate full level matrix
            prob_level_matrix = full_prob_levels_from_flat(
                classes_last, unique_classes_last, prob_
            )

            prob_ = prob_level_matrix

            # Extract correct indices to get probabilities of final labels
            prob_level = []
            for i in range(0, len(idx_curr_labels)):
                prob_level.append(prob_level_matrix[i, idx_curr_labels][i])

            # Store new probability
            df_prob_level = pd.DataFrame(prob_level)
            df_prob_level.index = ytrue

            df = pd.concat([df, df_prob_level], axis=1)
    if balanced:
        return (df, Labels_c_true, Labels_c_pred)
    else:
        return (df, Labels_c_true, Labels_c_pred, conversion_d)


def reject_flat_not_greedy(preds, prob, thresh, balanced=True):
    """Manual rejection of the labels for flat classificatiuon

    Parameters
    ----------
    preds : list
        predicted class labels
    prob : pd.DataFrame
        probability df with recorded proabbilities for every sublabel
    thresh : float
        rejection_threshold

    Returns
    -------
    list,
        predicted labels, possibly rejected

    list,
        probabilities corresponding to predicted labels

    """
    new_labels = []
    final_prob = []

    for j in range(0, len(preds)):
        p = preds[j]

        # The prob df is made with adapted unbalanced labels, so undo this
        if balanced:
            probs_ = prob.iloc[j, :].tolist()
        else:
            if len(prob.iloc[j, :].tolist()) != len(p.split(";")):
                probs_ = prob.iloc[j, :].tolist()[0 : len(p.split(";"))]
            else:
                probs_ = prob.iloc[j, :].tolist()

        i = 1
        curr_p = probs_[0]
        curr_label = p
        while curr_p < thresh:  # NOTE:  Check if this is correct
            curr_label = ";".join(p.split(";")[:-1])
            if curr_label == "root":
                break
            p = ";".join(p.split(";")[:-1])
            curr_p = probs_[i]

            i += 1

        new_labels.append(curr_label)
        final_prob.append(curr_p)

    return (new_labels, final_prob)


def Accuracy_Rejection_flat(clf, X, ytrue, predictions, probabilities,  balanced=False, scores_=False, step=0.01):
    steps = np.arange(0, 1 + step, step)
    acc_del = np.zeros(len(steps))
    acc_int = np.zeros(len(steps))
    lp = []
    lt = []
    p = []

    classes = clf.classes_

    # Loop over steps
    for i in range(0, len(steps)):
        t = steps[i]

        # recreate probs_hier_flat
        if balanced:
            prob_H, ytrue_, pred_ = hierprob_from_flat_non_greedy(
                probabilities, classes, predictions, ytrue, balanced=balanced
            )
            if pred_ != predictions != 0:
                raise ValueError("Something went wrong balanced")
            elif ytrue != ytrue_:
                raise ValueError("Something went wrong balanced 2")
        else:
            print('p',predictions)
            print('c',classes)
            print('y', ytrue)
            prob_H, ytrue_c, preds_c, conversion_d = hierprob_from_flat_non_greedy(
                probabilities, classes, predictions, ytrue
            )

            # converted unbalanced labels, recheck if ok

            preds_c = [
                conversion_d[p] if p in conversion_d.keys() else p for p in preds_c
            ]
            ytrue_c = [
                conversion_d[y] if y in conversion_d.keys() else y for y in ytrue_c
            ]

            if predictions != preds_c != 0:
                raise ValueError("Something went wrong unbalanced")
            elif ytrue != ytrue_c:
                raise ValueError("Something went wrong unbalanced 2")

        # for preds, check if prob[0] < thresh, if so check until prob > and remake pred
        pred, probs = reject_flat_not_greedy(predictions, prob_H, thresh=t)
        # Calculate accuracy
        acc_del[i] = accuracy_score_reject_delete(ytrue, pred)
        acc_int[i] = accuracy_score_reject_intermediate(ytrue, pred)
        # rejection percentage
        perc, lengths_pred, lengths_ytrue = find_percentage_reject(ytrue, pred)

        # calculate lengths
        p.append(perc)
        lp.append(lengths_pred)
        lt.append(lengths_ytrue)

    return (acc_del, acc_int, p, lp, lt, steps)


# Hierarchical Accuracy rejection specific functions
def Accuracy_Rejection(
    clf,
    X,
    y,
    step=0.01,
    greedy_=True,
):
    """_summary_

    Plots and or saves accuracy scores based on rejection thresholds.
    Here rejection is not explicitely performed. Rejected labels are just not included in the analysis

    Parameters
    ----------
    X : test dataT
    y : test labels
    greedy_: Boolean, optional
    step : float, optional
        steps in between rejection thresholds, by default 0.01
    """
    if isinstance(y, np.ndarray):  # happens when sparse
        y = y.flatten().tolist()
    if isinstance(y, pd.DataFrame):
        y = y[0].values.tolist()
    if isinstance(y, pd.Series):
        y = y.values.tolist()
    step = 0.01
    steps = np.arange(0, 1 + step, step)
    acc_del = np.zeros(len(steps))
    acc_int = np.zeros(len(steps))
    lp = []
    lt = []
    p = []
    for i in range(0, len(steps)):
        t = steps[i]
        predicted, probs = clf.predict(X, reject_thr=t, greedy=greedy_)
        acc_del[i] = accuracy_score_reject_delete(y, predicted)
        acc_int[i] = accuracy_score_reject_intermediate(y, predicted)
        perc, lengths_pred, lengths_ytrue = find_percentage_reject(y, predicted)
        p.append(perc)
        lp.append(lengths_pred)
        lt.append(lengths_ytrue)
    return (acc_del, acc_int, p , lp, lt, steps)

def _AR_parallel(clf, X, y, steps, t, acc_del, acc_int, greedy_=True):
        predicted, probs = clf.predict(X, reject_thr=steps, greedy=greedy_)
        acc_del[t] = accuracy_score_reject_delete(y, predicted)
        acc_int[t] = accuracy_score_reject_intermediate(y, predicted)
        perc, lengths_pred, lengths_ytrue = find_percentage_reject(y, predicted)
        return (acc_del, acc_int, perc, lengths_pred, lengths_ytrue, t)

def Accuracy_Rejection_Parallel(
    clf,
    X,
    y,
    jobs,
    step=0.01,
    greedy_=True,
):
    """_summary_

    Plots and or saves accuracy scores based on rejection thresholds.
    Here rejection is not explicitely performed. Rejected labels are just not included in the analysis

    Parameters
    ----------
    X : test dataT
    y : test labels
    greedy_: Boolean, optional
    step : float, optional
        steps in between rejection thresholds, by default 0.01
    """
    if isinstance(y, np.ndarray):  # happens when sparse
        y = y.flatten().tolist()
    if isinstance(y, pd.DataFrame):
        y = y[0].values.tolist()
    if isinstance(y, pd.Series):
        y = y.values.tolist()
    step = 0.01
    steps = np.arange(0, 1 + step, step)
    acc_del = np.zeros(len(steps))
    acc_int = np.zeros(len(steps))
    
    print(y)
    print(X)
    print(steps)
    acc_del, acc_int, perc, lp, lt, steps = zip(*Parallel(n_jobs = jobs)(
        delayed(_AR_parallel)(clf, X, y, steps[t] ,t, acc_del, acc_int, greedy_) for t in range(0, len(steps))
    ))
    return (acc_del, acc_int, perc, lp, lt, steps)

# Evaluation functions
def Evaluate_AR_Flat(clf_list, Xtests, ytests, predictions, probabilities, b, scores):
    """Function to generate datapoints for accuracy-rejection curves with flat classification 
       The rejection threshold is varied with a stepsize of 0.01.

    Parameters
    ----------
    clf_list : list of scikit-learn classifiers
        Contains the trained classifiers on the test sets over the K folds.
    Xtests : list of matrices
        Contains the test data per fold
    ytests : list of lists
        Contains the actual labels of the test data per fold
    predictions : list of lists
        Contains the predictions per fold
    probabilities : list of matrices
        Contains the probabilities of the predictions per fold over all the classes
    b : boolean
        Is the hierarchy balanced?
    scores : boolean
        Does the trained scikit-learn classifier output scores or probabilities (with predict_proba)?

    Returns
    -------
    results: nested dictionary
        The following metrics are saved in a dictionary with key 'Try fold_number' for every fold: the accuracies (acc) and rejection percentage (perc:) for every rejection threshold, the rejection thresholds themselves (steps), 
        the actual values (ytest), the predictions (preds), the probabilities (probs) and the lengths of the predictions (lp) and actual values (lt) per rejection threshold
    """
    results = {}
    for i in range(0, len(clf_list)):
        p = clf_list[i].predict(Xtests[i])
        acc_del, acc_int, perc, lp, lt, steps = Accuracy_Rejection_flat(
            clf_list[i], Xtests[i], ytests[i], predictions[i], probabilities[i], balanced=b, scores_=scores
        )
        results["Try " + str(i + 1)] = {
            "acc_del": acc_del,
            "acc_int": acc_int,
            "steps": steps,
            "ytest": ytests[i],
            "preds": predictions[i],
            "perc": perc,
            "length pred": lp,
            "length ytrue": lt,
            "probs": probabilities[i],
        }
    return results


def Evaluate_AR(clf_list, Xtests, ytests, predictions, greedy=True):
    """Function to generate datapoints for accuracy-rejection curves with hierarchical classification 
       The rejection threshold is varied with a stepsize of 0.01.

    Parameters
    ----------
    clf_list : list of scikit-learn classifiers
        Contains the trained classifiers on the test sets over the K folds.
    Xtests : list of matrices
        Contains the test data per fold
    ytests : list of lists
        Contains the actual labels of the test data per fold
    predictions : list of lists
        Contains the predictions per fold
    greedy : boolean, optional
        Perform greedy (True) or non-greedy (False) hierarchical classification, by default True

    Returns
    -------
    results: nested dictionary
        The following metrics are saved in a dictionary with key 'Try fold_number' for every fold: the accuracies (acc) and rejection percentage (perc:) for every rejection threshold, the rejection thresholds themselves (steps), 
        the actual values (ytest), the predictions (preds), the probabilities (probs) and the lengths of the predictions (lp) and actual values (lt) per rejection threshold
    """
    results = {}
    for i in range(0, len(clf_list)):
        print('starting accuracy', accuracy_score(ytests[i], predictions[i]))
        p, probs = clf_list[i].predict(Xtests[i])
        print('accuracy test', accuracy_score(ytests[i], p))
        acc_del, acc_int, perc, lp, lt, steps = Accuracy_Rejection(clf_list[i], Xtests[i], ytests[i], greedy_ = greedy)
        results["Try " + str(i + 1)] = {
            "acc_del": acc_del,
            "acc_int": acc_int,
            "steps": steps,
            "ytest": ytests[i],
            "preds": predictions[i],
            "perc:": perc,
            "length pred": lp,
            "length ytrue": lt,
            "probs": probs,
        }
    return results

from joblib import Parallel, delayed

def Evaluate_AR_parallel(clf_list, Xtests, ytests, predictions, all_jobs, greedy):
    """Function to generate datapoints for accuracy-rejection curves with hierarchical classification in a parallel manner 
       The rejection threshold is varied with a stepsize of 0.01.

    Parameters
    ----------
    clf_list : list of scikit-learn classifiers
        Contains the trained classifiers on the test sets over the K folds.
    Xtests : list of matrices
        Contains the test data per fold
    ytests : list of lists
        Contains the actual labels of the test data per fold
    predictions : list of lists
        Contains the predictions per fold
    all_jobs : int
        number of CPU cores to parallelize the calculations over
    greedy : boolean, optional
        Perform greedy (True) or non-greedy (False) hierarchical classification, by default True


    Returns
    -------
    results: nested dictionary
        The following metrics are saved in a dictionary with key 'Try fold_number' for every fold: the accuracies (acc) and rejection percentage (perc:) for every rejection threshold, the rejection thresholds themselves (steps), 
        the actual values (ytest), the predictions (preds), the probabilities (probs) and the lengths of the predictions (lp) and actual values (lt) per rejection threshold
    """
    results = {}
    for i in range(0, len(clf_list)):
        print('starting accuracy', accuracy_score(ytests[i], predictions[i]))
        p, probs = clf_list[i].predict(Xtests[i])
        print('accyracy test', accuracy_score(ytests[i], p))
        acc_del, acc_int, perc, lp, lt, steps = Accuracy_Rejection_Parallel(clf_list[i], Xtests[i], ytests[i], all_jobs, greedy_ = greedy)
        results["Try " + str(i + 1)] = {
            "acc_del": acc_del,
            "acc_int": acc_int,
            "steps": steps,
            "ytest": ytests[i],
            "preds": predictions[i],
            "perc:": perc,
            "length pred": lp,
            "length ytrue": lt,
            "probs": probs,
        }
    return results

