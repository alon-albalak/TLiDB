import json
import random

random.seed(13)
path = "TLiDB_DailyDialog/TLiDB_DailyDialog.json"
DD_tasks = [
      "emotion_recognition",
      "dialogue_act_classification",
      "topic_classification",
      "causal_emotion_span_extraction",
      "causal_emotion_entailment",
      "dialogue_nli",
      "dialogue_reasoning_span_extraction",
      "dialogue_reasoning_multiple_choice_span_selection",
      "dialogue_reasoning_commonsense_relation_prediction",
      "adversarial_response_selection"
    ]

utterance_level_tasks = ["emotion_recognition", "dialogue_act_classification"]
single_label_dialogue_level_tasks = ["topic_classification"]
list_label_dialogue_level_tasks = ["causal_emotion_span_extraction", "causal_emotion_entailment", "dialogue_nli", "dialogue_reasoning_commonsense_relation_prediction"]
qa_tasks = ["dialogue_reasoning_span_extraction"]
mc_tasks = ["dialogue_reasoning_multiple_choice_span_selection"]
response_selection_tasks = ["adversarial_response_selection"]

def sample_from_id_file(percent, ids=None, id_file=None):
    """
    Sample from a file of dialogue ids
    """
    if id_file:
        with open(id_file, 'r') as f:
            ids = f.read().splitlines()
    num_ids = len(ids)
    num_samples = int(num_ids * percent)
    sampled_ids = random.sample(ids, num_samples)
    return sampled_ids

def get_task_samples_per_split_by_dialogue_id(train_ids, dev_ids, test_ids, path=path):
    """
    Get the number of samples per task per split for each dialogue
    """
    tasks_by_split = {split:{task:0 for task in DD_tasks} for split in ["train", "dev", "test"]}
    splits = {"train":0, "dev":0, "test":0}
    ids_per_split = {split:[] for split in ["train", "dev", "test"]}
    ids_per_task_per_split = {split:{task:[] for task in DD_tasks} for split in ["train", "dev", "test"]}
    with open(path, 'r') as f:
        dataset = json.load(f)
    data = dataset['data']
    for dialogue in data:
        if dialogue['dialogue_id'] in train_ids:
            split = "train"
        elif dialogue['dialogue_id'] in dev_ids:
            split = "dev"
        elif dialogue['dialogue_id'] in test_ids:
            split = "test"
        else:
            continue
        splits[split] += 1
        ids_per_split[split].append(dialogue['dialogue_id'])
        for turn in dialogue['dialogue']:
            for task in utterance_level_tasks:
                if task in turn:
                    tasks_by_split[split][task] += 1
        for task in dialogue:
            if task in single_label_dialogue_level_tasks:
                tasks_by_split[split][task] += 1
            if task in list_label_dialogue_level_tasks:
                tasks_by_split[split][task] += len(dialogue[task])
                ids_per_task_per_split[split][task].append(dialogue['dialogue_id'])
            if task in qa_tasks:
                for qa in dialogue[task]:
                    tasks_by_split[split][task] += len(qa['qas'])
                    ids_per_task_per_split[split][task].append(dialogue['dialogue_id'])
            if task in mc_tasks:
                tasks_by_split[split][task] += len(dialogue[task]['mcqs'])
                ids_per_task_per_split[split][task].append(dialogue['dialogue_id'])
            if task in response_selection_tasks:
                for group in dialogue[task]:
                    tasks_by_split[split][task] += len(group['positive_responses'])
                    ids_per_task_per_split[split][task].append(dialogue['dialogue_id'])

    for split, tasks in tasks_by_split.items():
        print(f"{split}")
        for task, num_samples in tasks.items():
            print(f"\t{task}: {num_samples}")
    print(splits)


# subsample to get few-shot samples
def get_few_shot_samples(percent, id_file):
    subsampled_ids = sample_from_id_file(percent=percent, id_file=id_file)
    return subsampled_ids

few_shot_save_path="TLiDB_DailyDialog/TTiDB_{}_percent_few_shot_{}_ids.txt"

for percent in [0.1, 0.05, 0.01]:
    train_ids, dev_ids, test_ids = [], [], []
    for split in ["train", "dev"]:
        id_file="TLiDB_DailyDialog/TTiDB_{}_ids.txt".format(split)
        few_shot_ids = get_few_shot_samples(percent=percent, id_file=id_file)
        if split == "train":
            train_ids = few_shot_ids
        elif split == "dev":
            dev_ids = few_shot_ids
        with open(few_shot_save_path.format(percent, split), 'w') as f:
            f.writelines([f"{id}\n" for id in sorted(few_shot_ids, key=lambda x: int(x.split("-")[-1]))])
    print(f"FEW SHOT SAMPLES FOR {percent}")
    get_task_samples_per_split_by_dialogue_id(train_ids, dev_ids, test_ids, "TLiDB_DailyDialog/TLiDB_DailyDialog.json")
