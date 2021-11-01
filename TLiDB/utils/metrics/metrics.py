import numpy as np
import torch
from TLiDB.utils.utils import numel

class Metric:
    """
    Parent class for metrics
    """
    def __init__(self,name):
        self._name = name

    def _compute(self, y_pred, y_true):
        """
        Helper function for computing the metric.
        Subclasses should implement this.
        Args:
            - y_pred (Tensor): Predicted targets or model output
            - y_true (Tensor): True targets
        Output:
            - metric (0-dim tensor): metric
        """
        return NotImplementedError

    @property
    def name(self):
        """
        Metric name.
        Used to name the key in the results dictionaries returned by the metric.
        """
        return self._name

    @property
    def agg_metric_field(self):
        """
        The name of the key in the results dictionary returned by Metric.compute().
        This should correspond to the aggregate metric computed on all of y_pred and y_true,
        in contrast to a group-wise evaluation.
        """
        return f'{self.name}_all'

    def compute(self, y_pred, y_true, return_dict=True):
        """
        Computes metric. This is a wrapper around _compute.
        Args:
            - y_pred (Tensor): Predicted targets or model output
            - y_true (Tensor): True targets
            - return_dict (bool): Whether to return the output as a dictionary or a tensor
        Output (return_dict=False):
            - metric (0-dim tensor): metric. If the inputs are empty, returns tensor(0.)
        Output (return_dict=True):
            - results (dict): Dictionary of results, mapping metric.agg_metric_field to avg_metric
        """
        if numel(y_true) == 0:
            agg_metric = torch.tensor(0., device=y_true.device)
        else:
            agg_metric = self._compute(y_pred, y_true)
        if return_dict:
            results = {
                self.agg_metric_field: agg_metric.item()
            }
            return results
        else:
            return agg_metric


class ElementwiseMetric(Metric):
    """
    Averages.
    """
    def _compute_element_wise(self, y_pred, y_true):
        """
        Helper for computing element-wise metric, implemented for each metric
        Args:
            - y_pred (Tensor): Predicted targets or model output
            - y_true (Tensor): True targets
        Output:
            - element_wise_metrics (Tensor): tensor of size (batch_size, )
        """
        raise NotImplementedError

    def _compute(self, y_pred, y_true):
        """
        Helper function for computing the metric.
        Args:
            - y_pred (Tensor): Predicted targets or model output
            - y_true (Tensor): True targets
        Output:
            - avg_metric (0-dim tensor): average of element-wise metrics
        """
        element_wise_metrics = self._compute_element_wise(y_pred, y_true)
        avg_metric = element_wise_metrics.mean()
        return avg_metric

    @property
    def agg_metric_field(self):
        """
        The name of the key in the results dictionary returned by Metric.compute().
        """
        return f'{self.name}_avg'

    def compute_element_wise(self, y_pred, y_true, return_dict=True):
        """
        Computes element-wise metric
        Args:
            - y_pred (Tensor): Predicted targets or model output
            - y_true (Tensor): True targets
            - return_dict (bool): Whether to return the output as a dictionary or a tensor
        Output (return_dict=False):
            - element_wise_metrics (Tensor): tensor of size (batch_size, )
        Output (return_dict=True):
            - results (dict): Dictionary of results, mapping metric.name to element_wise_metrics
        """
        element_wise_metrics = self._compute_element_wise(y_pred, y_true)
        batch_size = y_pred.size()[0]
        assert element_wise_metrics.dim() == 1 and element_wise_metrics.numel() == batch_size

        if return_dict:
            return {self.name: element_wise_metrics}
        else:
            return element_wise_metrics
