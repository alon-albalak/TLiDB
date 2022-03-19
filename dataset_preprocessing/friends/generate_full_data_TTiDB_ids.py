import json
import random
random.seed(0)

path = "TLiDB_Friends/TLiDB_Friends.json"
Friends_tasks = [
    "emory_emotion_recognition",
    "reading_comprehension",
    "character_identification",
    "question_answering",
    "personality_detection",
    "relation_extraction",
    "MELD_emotion_recognition"
]


def get_num_task_samples_Friends(path=path):
    tasks = {task: 0 for task in Friends_tasks}
    with open(path, "r") as f:
        dataset = json.load(f)
    data = dataset["data"]

    tasks_per_dialogue = {}

    for dialogue in data:
        num_tasks_per_dialogue = 0
        em_found = False
        ci_found = False
        meld_found = False
        for turn in dialogue['dialogue']:
            if "emory_emotion_recognition" in turn:
                tasks["emory_emotion_recognition"] += 1
                em_found = True
            if "character_identification" in turn:
                tasks["character_identification"] += len(turn['character_identification'])
                ci_found = True
            if "MELD_emotion_recognition" in turn:
                tasks['MELD_emotion_recognition'] += 1
                meld_found = True
        if em_found:
            num_tasks_per_dialogue += 1
        if ci_found:
            num_tasks_per_dialogue += 1
        if meld_found:
            num_tasks_per_dialogue += 1
        if "reading_comprehension" in dialogue:
            tasks["reading_comprehension"] += len(dialogue['reading_comprehension'])
            num_tasks_per_dialogue += 1
        if "question_answering" in dialogue:
            for qa in dialogue['question_answering']:
                tasks["question_answering"] += len(qa['qas'])
            num_tasks_per_dialogue += 1
        if "personality_detection" in dialogue:
            tasks["personality_detection"] += len(dialogue['personality_detection'])
            num_tasks_per_dialogue += 1
        if "relation_extraction" in dialogue:
            for subdialogue in dialogue['relation_extraction']:
                for triple in subdialogue['relation_triples']:
                    tasks['relation_extraction'] += len(triple['relations'])
            num_tasks_per_dialogue += 1
        if num_tasks_per_dialogue not in tasks_per_dialogue:
            tasks_per_dialogue[num_tasks_per_dialogue] = 0
        tasks_per_dialogue[num_tasks_per_dialogue] += 1

    return tasks, tasks_per_dialogue

def split_dialogues(path=path, percent_dev=0.15, percent_test=0.15):
    with open(path, "r") as f:
        dataset = json.load(f)
    data = dataset["data"]

    ids = [dialogue['dialogue_id'] for dialogue in data]
    random.shuffle(ids)
    test_ids = ids[:int(len(ids) * percent_test)]
    dev_ids = ids[int(len(ids) * percent_test):int(len(ids) * (percent_test + percent_dev))]
    train_ids = ids[int(len(ids) * (percent_test + percent_dev)):]
    return train_ids, dev_ids, test_ids

def get_task_samples_per_split_by_dialogue_id(train_ids, dev_ids, test_ids, path=path):
    tasks_by_split = {split:{task: 0 for task in Friends_tasks} for split in ["train", "dev", "test"]}
    splits = {"train": 0, "dev": 0, "test": 0}

    with open(path, "r") as f:
        dataset = json.load(f)
    data = dataset["data"]
    for dialogue in data:
        if dialogue['dialogue_id'] in train_ids:
            split = "train"
        elif dialogue['dialogue_id'] in dev_ids:
            split = "dev"
        elif dialogue['dialogue_id'] in test_ids:
            split = "test"

        splits[split] += 1

        for turn in dialogue['dialogue']:
            if "emory_emotion_recognition" in turn:
                tasks_by_split[split]["emory_emotion_recognition"] += 1
            if "character_identification" in turn:
                tasks_by_split[split]['character_identification'] += len(turn['character_identification'])
            if "MELD_emotion_recognition" in turn:
                tasks_by_split[split]['MELD_emotion_recognition'] += 1
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


samples_per_task, tasks_per_dialogue = get_num_task_samples_Friends()
print(f"Samples per task: {samples_per_task}")
print(f"Tasks per dialogue: {sorted(tasks_per_dialogue.items())}")

train_ids, dev_ids, test_ids = split_dialogues()
get_task_samples_per_split_by_dialogue_id(train_ids, dev_ids, test_ids)
with open("TLiDB_Friends/TTiDB_train_ids.txt", "w") as f:
    f.writelines([f"{id}\n" for id in sorted(train_ids)])
with open("TLiDB_Friends/TTiDB_dev_ids.txt", "w") as f:
    f.writelines([f"{id}\n" for id in sorted(dev_ids)])
with open("TLiDB_Friends/TTiDB_test_ids.txt", "w") as f:
    f.writelines([f"{id}\n" for id in sorted(test_ids)])