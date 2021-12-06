import re
import string
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

class StringMetric:
    """
    Parent class for string metrics
    """
    def __init__(self,name,ignore_phrases):
        self._name = name
        self._ignore_phrases = [self._normalize_answer(text, string.punctuation, '') for text in ignore_phrases]

    def _compute(self, y_pred, y_true):
        """
        Helper function for computing the metric.
        Subclasses should implement this.
        Args:
            - y_pred (List of str): Predicted targets or model output
            - y_true (List of str): True targets
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
    def ignore_phrases(self):
        """
        List of phrases to ignore when computing the metric.
        """
        return self._ignore_phrases

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
            - y_pred (List of str): Predicted targets or model output
            - y_true (List of str): True targets
            - return_dict (bool): Whether to return the output as a dictionary or a tensor
        Output (return_dict=False):
            - metric (0-dim tensor): metric. If the inputs are empty, returns tensor(0.)
        Output (return_dict=True):
            - results (dict): Dictionary of results, mapping metric.agg_metric_field to avg_metric
        """
        if numel(y_true) == 0:
            agg_metric = torch.tensor(0., device=y_true.device)
        else:
            y_pred = [self._normalize_answer(text, string.punctuation, '') for text in y_pred]
            y_pred = [text if not any([ignore_phrase in text for ignore_phrase in self.ignore_phrases]) else "" for text in y_pred]
            y_true = [self._normalize_answer(text, string.punctuation, '') for text in y_true]
            y_true = [text if not any([ignore_phrase in text for ignore_phrase in self.ignore_phrases]) else "" for text in y_true]

            agg_metric = self._compute(y_pred, y_true)
        if return_dict:
            results = {
                self.agg_metric_field: agg_metric
            }
            return results
        else:
            return agg_metric

    def _normalize_answer(self, text, punc_chars, punc_repl):
        """
        Lower text and remove punctuation, articles and extra whitespace.
        Shamelessly copied from https://github.com/google-research/text-to-text-transfer-transformer/blob/220e43384912392302c34aaea9398dc5d66d975b/t5/evaluation/qa_utils.py#L29
        """

        def remove_articles(s):
            return re.sub(r"\b(a|an|the)\b", " ", s)

        def replace_punctuation(s):
            to_replace = set(punc_chars)
            return "".join(punc_repl if ch in to_replace else ch for ch in s)

        def white_space_fix(s):
            return " ".join(s.split())

        text = text.lower()
        text = replace_punctuation(text)
        text = remove_articles(text)
        text = white_space_fix(text)

        return text

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
