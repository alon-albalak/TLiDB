from .metrics import Metric, ElementwiseMetric
import sklearn.metrics

def multiclass_logits_to_pred(logits):
    """
    Takes multi-class logits of size (batch_size, ..., n_classes) and returns predictions
    by taking an argmax at the last dimension
    """
    assert logits.dim() > 1
    return logits.argmax(-1)

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
        super.__init__(name=name)

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