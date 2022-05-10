from .all_metrics import MetricGroup

def get_metric_computer(metrics, **kwargs):
    return MetricGroup(metrics, **kwargs)