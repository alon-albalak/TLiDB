import os
import csv
import sys

BASE_SOURCE_PATH="{}_{}.{}_seed.{}/{}"
BASE_COTRAINED_PATH="{}_{}_seed.{}/{}"
TASKS={"DailyDialog":[
    # DailyDialog original tasks
    "emotion_recognition", "dialogue_act_classification", "topic_classification",
    # RECCON tasks
    "causal_emotion_span_extraction", "causal_emotion_entailment",
    # CIDER tasks
    "dialogue_nli", "dialogue_reasoning_span_extraction",
    "dialogue_reasoning_multiple_choice_span_selection",
    "dialogue_reasoning_commonsense_relation_prediction",
    # DailyDialog++ task
    "adversarial_response_selection"
    ],
    "Friends": [
        # emoryNLP tasks
        "emotion_recognition", "reading_comprehension", "character_identification",
        "question_answering", "personality_detection",
        # dialogRE task
        "relation_extraction",
        # MELD task
        "MELD_emotion_recognition"
    ]
}

def get_single_result(log_path):
    """
    Get results from single file
    """
    try:
        with open(os.path.join(log_path, "log.txt"), 'r') as f:
            lines = f.readlines()
    except:
        return

    metrics_start = None

    for i, line in enumerate(lines):
        if "Eval on test split" in line:
            metrics_start = i+1
            break
    
    if not metrics_start:
        return
    
    metrics = {}
    for line in lines[metrics_start:]:
        metric_line = line.strip()

        # contine searching until empty line
        if metric_line:
            if len(metric_line.split(": ")) == 2:
                metric, value = metric_line.split(": ")
                metrics[metric] = float(value)
        else:
            break

    return metrics

def get_single_seed_results(training_prefix, model, dataset, seed, fewshot_percent=None):
    """
    Get results for all tasks for a single seed

    Args:
        training_prefix (str): prefix of the training directory, eg. "PRETRAINED_0.1_FEWSHOT" or "COTRAINED"
        model (str): model name
        dataset (str): dataset name
        seed (int): seed number
        fewshot_percent (float): percentage of fewshot data if using fewshot
        tasks (list): list of tasks
    """
    tasks = TASKS[dataset]

    results = {train_type: {target_task: {source_task:{} for source_task in tasks} \
                                for target_task in tasks} \
                    for train_type in ["pretrained_fine-tuned"]}

    # for each task, get the metrics for train/test on source task
    # then get the metrics for train on source, and test on target task
    for source_task in tasks:
        source_prefix = training_prefix if not fewshot_percent else training_prefix + f"_{fewshot_percent}_FEWSHOT"
        source_path = BASE_SOURCE_PATH.format(source_prefix, dataset, source_task, seed, model)
        results['pretrained_fine-tuned'][source_task][source_task] = get_single_result(source_path)
        for target_task in tasks:
            if target_task != source_task:
                target_prefix = "FINETUNED" if not fewshot_percent else f"FINETUNED_{fewshot_percent}_FEWSHOT"
                base_target_path=f"{target_prefix}_{dataset}.{target_task}_seed.{seed}"
                target_path = os.path.join(source_path,base_target_path)
                results['pretrained_fine-tuned'][target_task][source_task] = get_single_result(target_path)

    return results

def get_single_seed_results_cotrained(training_prefix, model, dataset, seed, fewshot_percent=None):
    """
    Get results for all tasks for a single seed

    Args:
        training_prefix (str): prefix of the training directory, eg. "PRETRAINED_0.1_FEWSHOT" or "COTRAINED"
        model (str): model name
        dataset (str): dataset name
        seed (int): seed number
        fewshot_percent (float): percentage of fewshot data if using fewshot
        tasks (list): list of tasks
    """
    tasks = TASKS[dataset]

    results = {train_type: {target_task: {source_task:{} for source_task in tasks} \
                                for target_task in tasks} \
                    for train_type in ["cotrained", "fine-tuned"]}

    # for each task, get the metrics for train/test on source task
    # then get the metrics for train on source, and test on target task
    for source_task in tasks:
        source_prefix = training_prefix if not fewshot_percent else training_prefix + f"_{fewshot_percent}_FEWSHOT"
        source_path = BASE_SOURCE_PATH.format(source_prefix, dataset, source_task, seed, model)
        source_path = source_path.replace("COTRAINED", "PRETRAINED")
        base_result = get_single_result(source_path)
        results['cotrained'][source_task][source_task] = base_result
        results['fine-tuned'][source_task][source_task] = base_result
        # results['cotrained'][source_task][source_task] = {}
        # results['fine-tuned'][source_task][source_task] = {}

        for target_task in tasks:
            if target_task != source_task:
                datasets = f"{dataset}.{source_task}_{dataset}.{target_task}"
                # gather results from cotraining alone
                cotrain_target_prefix = training_prefix if not fewshot_percent else training_prefix + f"_{fewshot_percent}_FEWSHOT"
                cotrain_target_path = BASE_COTRAINED_PATH.format(cotrain_target_prefix, datasets, seed, model)
                results['cotrained'][target_task][source_task] = get_single_result(cotrain_target_path)

                # gather results from fine-tuning after cotraining
                target_prefix = "FINETUNED" if not fewshot_percent else f"FINETUNED_{fewshot_percent}_FEWSHOT"
                base_target_path=f"{target_prefix}_{dataset}.{target_task}_seed.{seed}"
                finetune_target_path = os.path.join(cotrain_target_path,base_target_path)
                results['fine-tuned'][target_task][source_task] = get_single_result(finetune_target_path)

    return results

def print_results(results):
    """
    Print results to terminal
    """
    for k, v in results.items():
        print(k)
        for k1, v1 in v.items():
            print(f"\t{k1}: {v1}")
        print()

def convert_results_to_table(results, aggregation="average"):
    """
    Convert results to table

    Args:
        results (dict): results dictionary
        aggregation (str): aggregation method, either average or sum
    """
    headers = []

    columns = []
    for target_task, source_tasks in results.items():
        headers.append(target_task)
        column = []
        for _, metrics in source_tasks.items():
            if metrics:
                aggregate_value = sum(metrics.values()) if aggregation == "sum" else sum(metrics.values())/len(metrics)
                aggregate_value = round(aggregate_value, 4)
            else:
                aggregate_value = "N/A"
            column.append(aggregate_value)
        columns.append(column)
    return columns,headers

def convert_columns_to_differences(columns):
    for i, column in enumerate(columns):
        for j, value in enumerate(column):
            if i != j:
                if value == "N/A" or columns[i][i] == "N/A":
                    columns[i][j] = "N/A"
                else:
                    columns[i][j] = round(value - columns[i][i],4)
    return columns

def save_results_as_csv(columns, headers, save_path):
    """
    Save results as table in csv format
    """

    rows = list(zip(*columns))
    with open(save_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["Source\Target Task"] + headers)
        for header, row in zip(headers, rows):
            writer.writerow([header, *row])

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python3 results_analysis.py <training_prefix> <model> <dataset> <seed> [<fewshot_percent>]")
        print("Example: python3 results_analysis.py logs_and_models/PRETRAINED t5-base DailyDialog 0.1")
        exit(1)

    training_prefix = sys.argv[1]
    model = sys.argv[2]
    dataset = sys.argv[3]
    seed = sys.argv[4]
    fewshot_percent = None if len(sys.argv) < 6 else float(sys.argv[5])

    if "COTRAINED" in training_prefix:
        results = get_single_seed_results_cotrained(training_prefix, model, dataset, seed, fewshot_percent)
    else:
        results = get_single_seed_results(training_prefix, model, dataset, seed, fewshot_percent)

    for result_type, result in results.items():
        print(f"{result_type} results:")
        print_results(result)

        columns, headers = convert_results_to_table(result)
        base_save_path = f"{training_prefix}_{dataset}_{model}_seed.{seed}_"
        if fewshot_percent:
            base_save_path += f"FEWSHOT.{fewshot_percent}_"
        save_results_as_csv(columns, headers, base_save_path+f"{result_type}_results.csv")
        columns = convert_columns_to_differences(columns)
        save_results_as_csv(columns, headers, base_save_path+f"{result_type}_results_differences.csv")

# Sample usages as Python module

# results = get_single_seed_results("/TLiDB/examples/logs_and_models/PRETRAINED", "t5-base", "DailyDialog", 42)
# columns, headers = convert_results_to_table(results)
# save_results_as_csv(columns, headers,"results.csv")

# columns = convert_columns_to_differences(columns)
# save_results_as_csv(columns, headers,"results_differences.csv")