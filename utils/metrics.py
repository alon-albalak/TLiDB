from sklearn.metrics import f1_score, accuracy_score

def f1(predictions, references, labels=None, pos_label=1,average="binary",sample_weight=None):
    """ F1 score is the harmonic mean of precision and recall, computed as
        F1= 2 * (precision * recall) / (precision + recall)

    predictions: Predicted labels, as returned by a model.
    references: Ground truth labels.
    labels: The set of labels to include when average != 'binary', and
        their order if average is None. Labels present in the data can
        be excluded, for example to calculate a multiclass average ignoring
        a majority negative class, while labels not present in the data will
        result in 0 components in a macro average. For multilabel targets,
        labels are column indices. By default, all labels in y_true and
        y_pred are used in sorted order.
    average: This parameter is required for multiclass/multilabel targets.
        If None, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:
            binary: Only report results for the class specified by pos_label.
                This is applicable only if targets (y_{true,pred}) are binary.
            micro: Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            macro: Calculate metrics for each label, and find their unweighted mean.
                This does not take label imbalance into account.
            weighted: Calculate metrics for each label, and find their average
                weighted by support (the number of true instances for each label).
                This alters ‘macro’ to account for label imbalance; it can result
                in an F-score that is not between precision and recall.
            samples: Calculate metrics for each instance, and find their average
                (only meaningful for multilabel classification).
    sample_weight: Sample weights.
    """
    score = f1_score(references, predictions, labels=labels, pos_label=pos_label, average=average, sample_weight=sample_weight)
    return {"f1":float(score) if score.size==1 else score}

def accuracy(predictions, references, normalize=True, sample_weight=None):
    """ Accuracy is the proportion of correct predictions among the total number of samples, computed as:
        Acc = (TP + TN) / (TP + FP + TN + FN)
    
    predictions: Predicted labels, as returned by a model.
    references: Ground truth labels.
    normalize: If False, return the number of correctly classified samples.
        Otherwise, return the fraction of correctly classified samples.
    sample_weight: Sample weights.
    """
    return {"accuracy":float(accuracy_score(references, predictions, normalize=normalize, sample_weight=sample_weight))}