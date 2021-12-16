from collections import Counter
from .metrics import Metric, StringMetric, ElementwiseMetric
import sklearn.metrics
import torch

class Accuracy(ElementwiseMetric):
    def __init__(self, prediction_fn=None, name=None):
        self.prediction_fn = prediction_fn
        if name is None:
            name = 'acc'
        super().__init__(name=name)

    def _compute_element_wise(self, y_pred, y_true):
        if self.prediction_fn is not None:
            y_pred = self.prediction_fn(y_pred)
        return (y_pred==y_true).float()

class F1(Metric):
    def __init__(self, prediction_fn=None, name=None, average='macro', labels=None):
        """
        Calculate F1 score
        Args:
            - prediction_fn: Function to convert y_pred into the same format as y_true (for example, convert logits to max index)
            - name (str): Name of the metric
            - average (str): one of ['binary', 'micro', 'macro', 'weighted', 'samples']
            - labels: The set of labels to include when average != 'binary'  (if None, will use all labels)
        """
        self.prediction_fn = prediction_fn
        self.average = average
        self.labels = labels
        if name is None:
            name = 'F1'
        if average is not None:
            name += f'-{self.average}'
        super().__init__(name=name)

    def _compute(self, y_pred,y_true):
        """
        Args:
            - y_pred: Predicted labels
            - y_true: Ground truth labels
        See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html for further documentation
        """
        if self.prediction_fn is not None:
            y_pred = self.prediction_fn(y_pred)
        score = sklearn.metrics.f1_score(y_true, y_pred, average=self.average, labels=self.labels)
        return torch.tensor(score)

class Precision(Metric):
    def __init__(self, prediction_fn=None, name=None, average='macro', labels=None):
        """
        Calculate Precision
        Args:
            - prediction_fn: Function to convert y_pred into the same format as y_true (for example, convert logits to max index)
            - name (str): Name of the metric
            - average (str): one of ['binary', 'micro', 'macro', 'weighted', 'samples']
            - labels: The set of labels to include when average != 'binary'  (if None, will use all labels)
        """
        self.prediction_fn = prediction_fn
        self.average = average
        self.labels = labels
        if name is None:
            name = 'Precision'
        if average is not None:
            name += f'-{self.average}'
        super().__init__(name=name)

    def _compute(self, y_pred,y_true):
        """
        Args:
            - y_pred: Predicted labels
            - y_true: Ground truth labels
        See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html for further documentation
        """
        if self.prediction_fn is not None:
            y_pred = self.prediction_fn(y_pred)
        score = sklearn.metrics.precision_score(y_true, y_pred, average=self.average, labels=self.labels)
        return torch.tensor(score)

class Recall(Metric):
    def __init__(self, prediction_fn=None, name=None, average='macro', labels=None):
        """
        Calculate Recall
        Args:
            - prediction_fn: Function to convert y_pred into the same format as y_true (for example, convert logits to max index)
            - name (str): Name of the metric
            - average (str): one of ['binary', 'micro', 'macro', 'weighted', 'samples']
            - labels: The set of labels to include when average != 'binary'  (if None, will use all labels)
        """
        self.prediction_fn = prediction_fn
        self.average = average
        self.labels = labels
        if name is None:
            name = 'Recall'
        if average is not None:
            name += f'-{self.average}'
        super().__init__(name=name)

    def _compute(self, y_pred,y_true):
        """
        Args:
            - y_pred: Predicted labels
            - y_true: Ground truth labels
        See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html for further documentation
        """
        if self.prediction_fn is not None:
            y_pred = self.prediction_fn(y_pred)
        score = sklearn.metrics.recall_score(y_true, y_pred, average=self.average, labels=self.labels)
        return torch.tensor(score)

class token_F1(StringMetric):
    def __init__(self, prediction_fn=None, name=None, average="macro", unanswerable_phrases=[]):
        """
        Calculate F1 score for token comparisons
        Args:
            - prediction_fn: Function to convert y_pred into the same format as y_true (for example, convert logits to max index)
            - name (str): Name of the metric
        """
        self.prediction_fn = prediction_fn
        self.average = average
        if name is None:
            name = 'token_F1'
        if average is not None:
            name += f'-{self.average}'
        super().__init__(name=name, unanswerable_phrases=unanswerable_phrases)
    
    def _compute(self, y_pred, y_true):
        """
        Args:
            - y_pred (List of str): Predicted labels
            - y_true (List of str): Ground truth labels
        """
        if self.prediction_fn is not None:
            y_pred = self.prediction_fn(y_pred)

        # Taken from https://github.com/google-research/text-to-text-transfer-transformer/blob/main/t5/evaluation/metrics.py
        def _get_token_f1_macro(y_pred, y_true):
            common_token_counts = (
                Counter(y_true) &
                Counter(y_pred))
            sum_common = sum(common_token_counts.values())
            if len(y_pred) == 0 or len(y_true) == 0:
                return int(y_pred == y_true)
            if sum_common == 0:
                return 0
            precision = 1.0 * sum_common / len(y_pred)
            recall = 1.0 * sum_common / len(y_true)
            f1 = (2 * precision * recall) / (precision + recall)
            return f1

        def _get_token_f1_micro(y_pred, y_true):
            common_token_counts = (
                Counter(y_true) &
                Counter(y_pred))
            sum_common = sum(common_token_counts.values())
            if len(y_pred) == 0 or len(y_true) == 0:
                return int(y_pred == y_true), len(y_pred), len(y_true)
            return sum_common, len(y_pred)-sum_common, len(y_true)-sum_common

        if self.average == "macro":
            f1s = []
            for p, t in zip(y_pred, y_true):
                f1s.append(_get_token_f1_macro(p.split(), t.split()))
            return torch.mean(torch.tensor(f1s))
            
        elif self.average == "micro":
            tp, fp, fn = 0, 0, 0
            for pred, true in zip(y_pred, y_true):
                tp_, fp_, fn_ = _get_token_f1_micro(pred.split(), true.split())
                tp += tp_
                fp += fp_
                fn += fn_
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)
            return torch.tensor(f1)
        else:
            raise ValueError(f"Unknown average: {self.average}")

class Exact_Match(StringMetric):
    def __init__(self, prediction_fn=None, name=None, unanswerable_phrases=[], ignore_unanswerable=False):
        """
        Calculate exact match score
        Args:
            - prediction_fn: Function to convert y_pred into the same format as y_true (for example, convert logits to max index)
            - name (str): Name of the metric
        """
        self.prediction_fn = prediction_fn
        if name is None:
            name = 'Exact_Match'
        if ignore_unanswerable:
            name += '_pos_only'
        super().__init__(name=name, unanswerable_phrases=unanswerable_phrases, ignore_unanswerable=ignore_unanswerable)

    def _compute(self, y_pred, y_true):
        """
        Args:
            - y_pred (List of str): Predicted labels
            - y_true (List of str): Ground truth labels
        """
        if self.prediction_fn is not None:
            y_pred = self.prediction_fn(y_pred)

        matches = [float(pred == true) for pred, true in zip(y_pred, y_true)]
        return torch.mean(torch.tensor(matches))

class MetricGroup:
    """
    A simple class to group metrics together
    """
    _string_to_class = {
        "f1":F1,
        "precision":Precision,
        "recall":Recall,
        "accuracy":Accuracy,
        "token_f1":token_F1,
        "exact_match":Exact_Match
    }
    def __init__(self, metrics, **kwargs):
        self.metrics = []
        for metric_str in metrics:
            metric_str = metric_str.lower()
            metric = self._string_to_class[metric_str]
            # allow for multiple variations of a metric
            # eg. F1-micro and F1-macro
            if metric_str in kwargs.keys():
                metric_kwargs = kwargs[metric_str]
                if isinstance(metric_kwargs, list):
                    assert(isinstance(metric_kwargs[0], dict)),"metric kwargs must be dict or list of dicts"
                    for m in metric_kwargs:
                        self.metrics.append(metric(**m))
                elif isinstance(metric_kwargs, dict):
                    self.metrics.append(metric(**metric_kwargs))
            else:
                self.metrics.append(metric())

    def compute(self, y_pred, y_true):
        results = {}
        results_str = ""
        for metric in self.metrics:
            results.update(metric.compute(y_pred, y_true))
            results_str += f'{metric.name}: {results[metric.agg_metric_field]:.4f}\n'
        return results, results_str