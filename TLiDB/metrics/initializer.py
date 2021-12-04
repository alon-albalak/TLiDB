from .all_metrics import MetricGroup

task_metrics = {
    "emotion_recognition": ["F1","Accuracy"],
    "intent_detection": ["F1","Accuracy"],
    "intent_classification": ["F1","Accuracy"],
}

def get_metric_computer(task_name):
    if task_name not in task_metrics:
        raise ValueError(f"{task_name} not found")
    return MetricGroup(task_metrics[task_name])