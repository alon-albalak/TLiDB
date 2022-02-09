from itertools import chain
from collections import Counter
from .metrics import Metric, StringMetric, ElementwiseMetric
import numpy as np
import sklearn.metrics
import torch

from sacrebleu.metrics import BLEU as SacreBLEU
import nltk
import bert_score
# import nlgeval

class binary_threshold():
    def __init__(self, threshold=0.5):
        self.threshold = threshold
    def __call__(self, y_pred):
        if isinstance(y_pred, torch.Tensor):
            return (y_pred > self.threshold).float()
        elif isinstance(y_pred, np.ndarray):
            return (y_pred > self.threshold).astype(np.float32)
        else:
            return (y_pred > self.threshold)

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

class MultiLabelF1(Metric):
    def __init__(self, prediction_fn=None, name=None, average='micro', labels=None):
        """
        Calculate F1 score for multi-label classification
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

class LRAP(Metric):
    def __init__(self, prediction_fn=None, name=None, labels=None):
        """
        Calculate a ranking-based average precision
        Args:
            - prediction_fn: Function to convert y_pred into the same format as y_true (for example, convert logits to max index)
            - name (str): Name of the metric
            - labels: the set of labels to include (if None, will include all labels)
        """
        self.prediction_fn = prediction_fn
        self.labels = labels
        if name is None:
            name = "LRAP"
        super().__init__(name=name)

    def _compute(self, y_pred, y_true):
        """
        Args:
            - y_pred: Predicted logits
            - y_true: Ground truth
        See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.label_ranking_average_precision_score.html for further documentation
        """
        if self.prediction_fn is not None:
            y_pred = self.prediction_fn(y_pred)
        
        if self.labels:
            # remove samples which do not have a desired label
            filtered_y_pred, filtered_y_true = [], []
            for pred, true in zip(y_pred, y_true):
                if any([true[l] != 0 for l in self.labels]):
                    if isinstance(pred, torch.Tensor):
                        pred = pred.numpy()
                    if isinstance(true, torch.Tensor):
                        true = true.numpy()
                    filtered_y_pred.append(pred)
                    filtered_y_true.append(true)
            y_pred = np.array(filtered_y_pred)
            y_true = np.array(filtered_y_true)
        
        score = sklearn.metrics.label_ranking_average_precision_score(y_true, y_pred)
        return torch.tensor(score)

class MRR(Metric):
    def __init__(self, prediction_fn=None, name=None, labels=None):
        """
        Calculate a variant of the mean reciprocal rank which considers all labels
            If there is only 1 ground truth label, this is equivalent to standard MRR
            For multi-label samples, this still allows for multiple
        Args:
            - prediction_fn: Function to convert y_pred into the same format as y_true (for example, convert logits to max index)
            - name (str): Name of the metric
            - labels: the set of labels to include (if None, will include all labels)
        """
        self.prediction_fn = prediction_fn
        self.labels = labels
        if name is None:
            name = "MRR"
        super().__init__(name=name)

    def _compute(self, y_pred, y_true):
        """
        Args:
            - y_pred: Predicted logits
            - y_true: Ground truth
        """
        
        if self.prediction_fn is not None:
            y_pred = self.prediction_fn(y_pred)
        
        if self.labels:
            # remove samples which do not have a desired label
            filtered_y_pred, filtered_y_true = [], []
            for pred, true in zip(y_pred, y_true):
                if any([true[l] != 0 for l in self.labels]):
                    if isinstance(pred, torch.Tensor):
                        pred = pred.numpy()
                    if isinstance(true, torch.Tensor):
                        true = true.numpy()
                    filtered_y_pred.append(pred)
                    filtered_y_true.append(true)
            y_pred = filtered_y_pred
            y_true = filtered_y_true
        
        reciprocal_ranks = []
        sorted_pred_idxs = np.argsort(-np.array(y_pred), axis=1)
        labels = [np.nonzero(l)[0] for l in y_true]

        for pred_idx, label in zip(sorted_pred_idxs, labels):
            found_labels = 0
            for rank, idx in enumerate(pred_idx):
                if idx in label:
                    reciprocal_ranks.append(1.0 / (rank - found_labels + 1))
                    found_labels += 1

        score = sum(reciprocal_ranks) / len(reciprocal_ranks)
        
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
            - y_pred (List of str OR List of List of str): Predicted labels
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

                if isinstance(t, str):
                    f1 = _get_token_f1_macro(p.split(), t.split())
                elif isinstance(t, list):
                    # if multiple ground truths, select the max
                    f1 = self._metric_max_over_ground_truths(_get_token_f1_macro, p.split(), [t_.split() for t_ in t])

                f1s.append(f1)
            return torch.mean(torch.tensor(f1s, dtype=torch.float))
            
        elif self.average == "micro":
            tp, fp, fn = 0, 0, 0
            for pred, true in zip(y_pred, y_true):

                if isinstance(true, str):
                    tp_, fp_, fn_ = _get_token_f1_micro(pred.split(), true.split())
                elif isinstance(true, list):
                    # if multiple ground truths, select the option with highest f1
                    best_tp, best_fp, best_fn, best_f1 = 0, 0, 0, 0
                    for t in true:
                        tp_, fp_, fn_ = _get_token_f1_micro(pred.split(), t.split())
                        f1_ = (2 * tp_) / (2 * tp_ + fp_ + fn_)
                        if f1_ > best_f1:
                            best_tp, best_fp, best_fn = tp_, fp_, fn_
                    tp_ = best_tp
                    fp_ = best_fp
                    fn_ = best_fn

                tp += tp_
                fp += fp_
                fn += fn_
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)
            return torch.tensor(f1, dtype=torch.float)
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

        def _get_exact_match(pred, true):
            return float(pred==true)

        multiple_ground_truths = isinstance(y_true[0],list)

        if multiple_ground_truths:
            matches = [self._metric_max_over_ground_truths(_get_exact_match, p, t) for p, t in zip(y_pred, y_true)]
        else:
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
        "multilabel_f1":MultiLabelF1,
        "label_ranking_average_precision": LRAP,
        "mean_reciprocal_rank": MRR,
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


####################################
### Evaluation Metrics for Response Generation ###
# - [x] BLEU-1, BLEU-2, BLEU-3, BLEU-4
# - [x] BertScore
# - [x] Distinct-N
# - [x] https://github.com/Maluuba/nlg-eval

# Priority list:
# 1. BLEUs, Distinct-N, Perplexity
# 2. BERT Score, cosine similarity (+ others from NLG Eval)
# 3. Meteor, ROUGE, CIDEr


# If possible, get Meteor, ROUGE-L, CIDEr from sources other than nlg-eval
# Add perplexity (loss) as metric

####################################

class BLEUs(StringMetric):
    def __init__(self, prediction_fn=None, name=None, unanswerable_phrases=[], ignore_unanswerable=False):
        """
        Calculate BLEU (specifically using SacreBLEU) unigram/bigram/trigram/4-grams scores
        Args:
            - prediction_fn: Function to convert y_pred into the same format as y_true (for example, convert logits to max index)
            - name (str): Name of the metric
        """
        self.bleu = SacreBLEU()
        self.prediction_fn = prediction_fn
        if name is None:
            name = 'BLEU'
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

        bleu_scores = self.bleu.corpus_score(y_pred, [[y] for y in y_true])
        return bleu_scores


class BertScore(StringMetric):
    def __init__(self, prediction_fn=None, name=None, unanswerable_phrases=[], ignore_unanswerable=False):
        """
        Calculate BertScore
        Args:
            - prediction_fn: Function to convert y_pred into the same format as y_true (for example, convert logits to max index)
            - name (str): Name of the metric
        """
        self.prediction_fn = prediction_fn
        if name is None:
            name = 'BertScore'
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

        P, R, F1 = bert_score.score(y_pred, y_true, lang='en', verbose=False)#, verbose=True, device='cuda')#FIXME: check if speficifying device here.
        return torch.mean(F1)


# class NLGEval(StringMetric):
#     def __init__(self, prediction_fn=None, name=None, unanswerable_phrases=[], ignore_unanswerable=False):
#         """
#         Calculate general natural language generation metrics, including: BLEU, METEOR, ROUGE-L, CIDEr, CosineSimilarity, GreedyMatching
#         Args:
#             - prediction_fn: Function to convert y_pred into the same format as y_true (for example, convert logits to max index)
#             - name (str): Name of the metric
#         """
#         self.prediction_fn = prediction_fn
#         if name is None:
#             name = 'NLGEval'
#         if ignore_unanswerable:
#             name += '_pos_only'
#         super().__init__(name=name, unanswerable_phrases=unanswerable_phrases, ignore_unanswerable=ignore_unanswerable)

#     def _compute(self, y_pred, y_true):
#         """
#         Args:
#             - y_pred (List of str): Predicted labels
#             - y_true (List of str): Ground truth labels
#         """
#         if self.prediction_fn is not None:
#             y_pred = self.prediction_fn(y_pred)

#         metrics_dict = nlgeval.compute_individual_metrics(y_true, y_pred)
#         return metrics_dict


class DistinctN(StringMetric):
    def __init__(self, prediction_fn=None, name=None, unanswerable_phrases=[], ignore_unanswerable=False):
        """
        Calculate distinct n-grams with a nltk word tokenizer
        Args:
            - prediction_fn: Function to convert y_pred into the same format as y_true (for example, convert logits to max index)
            - name (str): Name of the metric
        """
        self.prediction_fn = prediction_fn
        if name is None:
            name = 'DistinctN'
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

        y_pred = [nltk.tokenize.word_tokenizer(y) for y in y_pred]

        ngrams1 = list(chain(*[[gram for gram in nltk.ngrams(y, 1)] for y in y_pred]))
        ngrams2 = list(chain(*[[gram for gram in nltk.ngrams(y, 2)] for y in y_pred]))
        ngrams3 = list(chain(*[[gram for gram in nltk.ngrams(y, 3)] for y in y_pred]))

        metrics_dict = {
            'distinct-1': len(set(ngrams1)) / len(ngrams1),
            'distinct-2': len(set(ngrams2)) / len(ngrams2),
            'distinct-3': len(set(ngrams3)) / len(ngrams3),
        }

        return metrics_dict

