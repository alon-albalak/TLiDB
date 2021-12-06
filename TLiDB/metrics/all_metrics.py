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

class token_F1(StringMetric):
    def __init__(self, prediction_fn=None, name=None, ignore_phrases=[]):
        """
        Calculate F1 score for token comparisons
        Args:
            - prediction_fn: Function to convert y_pred into the same format as y_true (for example, convert logits to max index)
            - name (str): Name of the metric
        """
        self.prediction_fn = prediction_fn
        if name is None:
            name = 'token_F1'
        super().__init__(name=name, ignore_phrases=ignore_phrases)
    
    def _compute(self, y_pred, y_true):
        """
        Args:
            - y_pred (List of str): Predicted labels
            - y_true (List of str): Ground truth labels
        """
        if self.prediction_fn is not None:
            y_pred = self.prediction_fn(y_pred)

        # complicated, but maybe faster, version
        # Taken from https://github.com/google-research/text-to-text-transfer-transformer/blob/main/t5/evaluation/metrics.py
        # def _get_token_f1(y_pred, y_true):
        #     common_token_counts = (
        #         Counter(y_true) &
        #         Counter(y_pred))
        #     sum_common = sum(common_token_counts.values())
        #     if sum_common == 0:
        #         return 0
        #     precision = 1.0 * sum_common / len(y_pred)
        #     recall = 1.0 * sum_common / len(y_true)
        #     f1 = (2 * precision * recall) / (precision + recall)
        #     return f1
        # f1s = []
        # for p, t in zip(y_pred, y_true):
        #     f1s.append(_get_token_f1(p, t))
        
        # Visually simpler version
        tp, fp, fn = 0, 0, 0
        for pred, true in zip(y_pred, y_true):
            for pred_token in pred.split():
                if pred_token in true:
                    tp += 1
                else:
                    fp += 1
            for true_token in true.split():
                if true_token not in pred:
                    fn += 1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        return torch.tensor(f1)


class Exact_Match(StringMetric):
    def __init__(self, prediction_fn=None, name=None, ignore_phrases=[]):
        """
        Calculate exact match score
        Args:
            - prediction_fn: Function to convert y_pred into the same format as y_true (for example, convert logits to max index)
            - name (str): Name of the metric
        """
        self.prediction_fn = prediction_fn
        if name is None:
            name = 'Exact_Match'
        super().__init__(name=name, ignore_phrases=ignore_phrases)

    def _compute(self, y_pred, y_true):
        """
        Args:
            - y_pred (List of str): Predicted labels
            - y_true (List of str): Ground truth labels
        """
        if self.prediction_fn is not None:
            y_pred = self.prediction_fn(y_pred)
        # filter out pred/truth when both are empty string
        clean_y_true, clean_y_pred = [], []
        for true, pred in zip(y_true, y_pred):
            if true != '' or pred != '':
                clean_y_true.append(true)
                clean_y_pred.append(pred)
        matches = [float(pred == true) for pred, true in zip(clean_y_pred, clean_y_true)]
        return torch.mean(torch.tensor(matches))


class MetricGroup:
    """
    A simple class to group metrics together
    """
    _string_to_class = {
        "f1":F1,
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