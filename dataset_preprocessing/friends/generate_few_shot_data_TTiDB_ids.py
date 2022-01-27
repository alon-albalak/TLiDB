import json
import random

import statistics

random.seed(2530)

expected_10_percent = {
    "emotion_recognition":[863,207],
    "reading_comprehension":[950,208],
    "character_identification":[1692,387],
    "question_answering":[835,197],
    "personality_detection":[55,15],
    "relation_extraction":[530,130]
}


path = "TLiDB_Friends/TLiDB_Friends.json"
Friends_tasks = [
    "emotion_recognition",
    "reading_comprehension",
    "character_identification",
    "question_answering",
    "personality_detection",
    "relation_extraction"
]

def sample_from_id_file(percent, id_file=None):
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

def get_task_samples_per_split_by_dialogue_id(train_ids, dev_ids, path=path):
    tasks_by_split = {split:{task: 0 for task in Friends_tasks} for split in ["train", "dev"]}
    splits = {"train": 0, "dev": 0}

    with open(path, "r") as f:
        dataset = json.load(f)
    data = dataset["data"]
    for dialogue in data:
        if dialogue['dialogue_id'] in train_ids:
            split = "train"
        elif dialogue['dialogue_id'] in dev_ids:
            split = "dev"
        else:
            continue

        splits[split] += 1

        for turn in dialogue['dialogue']:
            if "emotion_recognition" in turn:
                tasks_by_split[split]["emotion_recognition"] += 1
            if "character_identification" in turn:
                tasks_by_split[split]['character_identification'] += 1
        if "reading_comprehension" in dialogue:
            tasks_by_split[split]['reading_comprehension'] += len(dialogue['reading_comprehension'])
        if "question_answering" in dialogue:
            for qa in dialogue['question_answering']:
                tasks_by_split[split]['question_answering'] += len(qa['qas'])
        if "personality_detection" in dialogue:
            tasks_by_split[split]['personality_detection'] += len(dialogue['personality_detection'])
        if "relation_extraction" in dialogue:
            for subdialogue in dialogue['relation_extraction']:
                for triple in subdialogue['relation_triples']:
                    tasks_by_split[split]['relation_extraction'] += len(triple['relations'])

    for split, tasks in tasks_by_split.items():
        print(f"{split}")
        for task, num_samples in tasks.items():
            print(f"\t{task}: {num_samples}")
    print(splits)



# subsample to get few-shot samples
def get_few_shot_samples(percent, id_file):
    subsampled_ids = sample_from_id_file(percent=percent, id_file=id_file)
    return subsampled_ids

few_shot_save_path = "TLiDB_Friends/TTiDB_{}_percent_few_shot_{}_ids.txt"

for percent in [0.1]:
    train_ids, dev_ids = [], []
    for split in ["train", "dev"]:
        id_file = "TLiDB_Friends/TTiDB_{}_ids.txt".format(split)
        few_shot_ids = get_few_shot_samples(percent=percent, id_file=id_file)
        if split == "train":
            train_ids = few_shot_ids
        elif split == "dev":
            dev_ids = few_shot_ids
        with open(few_shot_save_path.format(percent, split), "w") as f:
            f.writelines(f"{id}\n" for id in sorted(few_shot_ids))
    print(f"FEW SHOT SAMPLES FOR {percent}")
    get_task_samples_per_split_by_dialogue_id(train_ids, dev_ids)
