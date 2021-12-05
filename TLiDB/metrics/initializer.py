from .all_metrics import MetricGroup

def get_metric_computer(metrics):
    return MetricGroup(metrics)