from .metrics import Metric, ElementwiseMetric
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
    def __init__(self, prediction_fn=None, name=None, average='macro'):
        """
        Calculate F1 score
        Args:
            - prediction_fn: Function to convert y_pred into the same format as y_true (for example, convert logits to max index)
            - name (str): Name of the metric
            - average (str): one of ['binary', 'micro', 'macro', 'weighted', 'samples']
        For further documentation, see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
        """
        self.prediction_fn = prediction_fn
        self.average = average
        if name is None:
            name = 'F1'
            if average is not None:
                name += f'-{self.average}'
        super().__init__(name=name)

    def _compute(self, y_pred,y_true,labels=None):
        """
        Args:
            - y_pred: Predicted labels
            - y_true: Ground truth labels
            - labels: The set of labels to include when average != 'binary'  (if None, will use all labels)
        See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html for further documentation
        """
        if self.prediction_fn is not None:
            y_pred = self.prediction_fn(y_pred)
        score = sklearn.metrics.f1_score(y_true, y_pred, average=self.average, labels=labels)
        return torch.tensor(score)


class MetricGroup:
    """
    A simple class to group metrics together
    """
    _string_to_class = {
        "F1":F1,
        "Accuracy":Accuracy
    }
    def __init__(self, metrics):
        self.metrics = [self._string_to_class[metric]() for metric in metrics]

    def compute(self, y_pred, y_true):
        results = {}
        results_str = ""
        for metric in self.metrics:
            results.update(metric.compute(y_pred, y_true))
            results_str += f'{metric.name}: {results[metric.agg_metric_field]:.4f}\n'
        return results, results_str