import json
import random
import os

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

total_samples = {
    "emotion_recognition":102978,
    "dialogue_act_classification":102978,
    "topic_classification":13118,
    "causal_emotion_span_extraction":36324,
    "causal_emotion_entailment":36324,
    "dialogue_nli":5817,
    "dialogue_reasoning_span_extraction":1098,
    "dialogue_reasoning_multiple_choice_span_selection":2165,
    "dialogue_reasoning_commonsense_relation_prediction":4009,
    "adversarial_response_selection":57145
}


def get_num_task_samples_DD(path=path):
    tasks_by_split = {split:{task:0 for task in DD_tasks} for split in ["train", "dev", "test"]}
    splits = {"train":0, "dev":0, "test":0}
    ids_per_split = {split:[] for split in ["train", "dev", "test"]}
    ids_per_task_per_split = {split:{task:[] for task in DD_tasks} for split in ["train", "dev", "test"]}
    with open(path, 'r') as f:
        dataset = json.load(f)
    data = dataset['data']
    for dialogue in data:
        if "causal_emotion_span_extraction" in dialogue['dialogue_metadata']:
            split = dialogue['dialogue_metadata']['causal_emotion_span_extraction']['original_data_partition']
        else:
            split = dialogue['dialogue_metadata']['original_data_partition']
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
    # for split, tasks in tasks_by_split.items():
    #     print(f"{split}")
    #     for task, num_samples in tasks.items():
    #         print(f"\t{task}: {num_samples}")
    # print(splits)


    # ensure we haven't lost any samples in the rebalancing
    totals = {task:0 for task in DD_tasks}
    for split, tasks in tasks_by_split.items():
        for task, num_samples in tasks.items():
            totals[task] += num_samples
    for task, num_samples in totals.items():
        assert(num_samples == total_samples[task])

    for split in ids_per_split:
        with open(f"TLiDB_DailyDialog/{split}_ids.txt", 'w') as f:
            f.writelines([f"{id}\n" for id in ids_per_split[split]])

    for split in ids_per_task_per_split:
        for task in ids_per_task_per_split[split]:
            with open(f"TLiDB_DailyDialog/{split}_{task}_ids.txt", 'w') as f:
                f.writelines([f"{id}\n" for id in ids_per_task_per_split[split][task]])

def get_task_samples_per_split_by_dialogue_id_full_data(train_ids, dev_ids, test_ids, path=path):
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
            raise ValueError("Dialogue id not found in train, dev, or test set")
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


    # ensure we haven't lost any samples in the rebalancing
    totals = {task:0 for task in DD_tasks}
    for split, tasks in tasks_by_split.items():
        for task, num_samples in tasks.items():
            totals[task] += num_samples
    for task, num_samples in totals.items():
        assert(num_samples == total_samples[task])

    for split, tasks in tasks_by_split.items():
        print(f"{split}")
        for task, num_samples in tasks.items():
            print(f"\t{task}: {num_samples} - {num_samples/totals[task]:0.3f}")
    print(splits)

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

def move_ids_from_task_to_proposed_id_files(percents_to_dev, percents_to_test,tasks):
    with open(f"TLiDB_DailyDialog/train_ids.txt", 'r') as f:
        overall_train_ids = f.read().splitlines()
    with open(f"TLiDB_DailyDialog/dev_ids.txt", 'r') as f:
        overall_dev_ids = f.read().splitlines()
    with open(f"TLiDB_DailyDialog/test_ids.txt", 'r') as f:
        overall_test_ids = f.read().splitlines()

    for percent_to_dev, percent_to_test, task in zip(percents_to_dev, percents_to_test, tasks):
        print(task.upper())
        with open(f"TLiDB_DailyDialog/train_{task}_ids.txt", 'r') as f:
            train_ids = f.read().splitlines()
        with open(f"TLiDB_DailyDialog/dev_{task}_ids.txt", 'r') as f:
            dev_ids = f.read().splitlines()
        with open(f"TLiDB_DailyDialog/test_{task}_ids.txt", 'r') as f:
            test_ids = f.read().splitlines()
        
        print("STARTING DIALOGUES")
        print(len(train_ids))
        print(len(dev_ids))
        print(len(test_ids))

        failed = 0

        sampled_train_ids = sample_from_id_file(train_ids, percent_to_dev)
        print(f"{len(sampled_train_ids)} train ids to dev")
        for id in sampled_train_ids:
            if id in overall_train_ids:
                train_ids.remove(id)
                dev_ids.append(id)
                overall_train_ids.remove(id)
                overall_dev_ids.append(id)
            else:
                failed += 1

        sampled_train_ids = sample_from_id_file(train_ids, percent_to_test)
        print(f"{len(sampled_train_ids)} test ids to test")
        for id in sampled_train_ids:
            if id in overall_train_ids:
                train_ids.remove(id)
                test_ids.append(id)
                overall_train_ids.remove(id)
                overall_test_ids.append(id)
            else:
                failed += 1

        print("ADJUSTED DIALOGUES")
        print(len(train_ids))
        print(len(dev_ids))
        print(len(test_ids))

        print(f"{failed} FAILED")
    with open(f"TLiDB_DailyDialog/TTiDB_train_ids.txt", 'w') as f:
        f.writelines([f"{id}\n" for id in sorted(overall_train_ids, key=lambda x: int(x.split("-")[-1]))])
    with open(f"TLiDB_DailyDialog/TTiDB_dev_ids.txt", 'w') as f:
        f.writelines([f"{id}\n" for id in sorted(overall_dev_ids, key=lambda x: int(x.split("-")[-1]))])
    with open(f"TLiDB_DailyDialog/TTiDB_test_ids.txt", 'w') as f:
        f.writelines([f"{id}\n" for id in sorted(overall_test_ids, key=lambda x: int(x.split("-")[-1]))])

    get_task_samples_per_split_by_dialogue_id_full_data(overall_train_ids, overall_dev_ids, overall_test_ids, "TLiDB_DailyDialog/TLiDB_DailyDialog.json")


# subsample to get few-shot samples
def get_few_shot_samples(percent, id_file):
    subsampled_ids = sample_from_id_file(percent=percent, id_file=id_file)
    return subsampled_ids

get_num_task_samples_DD()

tasks = ["dialogue_reasoning_multiple_choice_span_selection", "adversarial_response_selection"]
percents_to_dev = [0.05, 0.06]
percents_to_test = [0.1, 0.09]
random.seed(10)
move_ids_from_task_to_proposed_id_files(percents_to_dev=percents_to_dev,percents_to_test=percents_to_test,tasks=tasks)

for file in os.listdir("TLiDB_DailyDialog"):
    if not any([prefix in file for prefix in ["TLiDB", "TTiDB"]]):
        os.remove(os.path.join("TLiDB_DailyDialog",file))

# subsample from full data to get few-shot dialogues
# we need to resample multiple times in order to get splits with appropriate samples per task

few_shot_save_path="TLiDB_DailyDialog/TTiDB_{}_percent_few_shot_{}_ids.txt"

for percent in [0.1, 0.05, 0.01]:
    for split in ["train", "dev"]:
        id_file="TLiDB_DailyDialog/TTiDB_{}_ids.txt".format(split)
        few_shot_ids = get_few_shot_samples(percent=percent, id_file=id_file)
        with open(few_shot_save_path.format(percent, split), 'w') as f:
            f.writelines([f"{id}\n" for id in sorted(few_shot_ids, key=lambda x: int(x.split("-")[-1]))])
# first sampling of 5% gives 0 samples for some tasks
for split in ["train","dev"]:
    id_file="TLiDB_DailyDialog/TTiDB_{}_ids.txt".format(split)
    few_shot_ids = get_few_shot_samples(percent=0.05, id_file=id_file)
    with open(few_shot_save_path.format(0.05, split), 'w') as f:
        f.writelines([f"{id}\n" for id in sorted(few_shot_ids, key=lambda x: int(x.split("-")[-1]))])