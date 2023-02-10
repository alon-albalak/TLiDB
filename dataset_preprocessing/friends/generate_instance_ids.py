import json

TASK_TYPE_MAP={
    "emory_emotion_recognition": "utt_level_classification",
    "MELD_emotion_recognition": "utt_level_classification",
    "reading_comprehension": "character_span_extraction",
    "character_identification": "character_identification",
    "question_answering": "span_extraction",
    "personality_detection": "personality_detection",
    "relation_extraction": "relation_extraction"
}


def generate_instance_ids(dataset):
    for datum in dataset['data']:
        for task in datum['dialogue_metadata']:
            task_type = TASK_TYPE_MAP[task]
            if task_type == "utt_level_classification":
                for turn in datum['dialogue']:
                    if task in turn:
                        instance_id = f"{datum['dialogue_id']}_t{turn['turn_id']}"
                        turn[task] = {"label": turn[task], "instance_id": instance_id}
            elif task_type == "character_span_extraction":
                for i, qa in enumerate(datum[task]):
                    instance_id = f"{datum['dialogue_id']}_qa{i}"
                    qa['instance_id'] = instance_id
            elif task_type == "character_identification":
                for turn in datum['dialogue']:
                    if task in turn:
                        for i, entity_mention in enumerate(turn[task]):
                            instance_id = f"{datum['dialogue_id']}_t{turn['turn_id']}_mention{i}"
                            entity_mention['instance_id'] = instance_id
            elif task_type == "span_extraction":
                for qas in datum[task]:
                    for qa in qas['qas']:
                        qa['instance_id'] = qa['id']
                        del qa['id']
            elif task_type == "personality_detection":
                for sample in datum[task]:
                    instance_id = f"{datum['dialogue_id']}_{sample['focus_speaker'].replace(' ','-')}"
                    sample['instance_id'] = instance_id
            elif task_type == "relation_extraction":
                for sample in datum[task]:
                    for triple in sample['relation_triples']:
                        instance_id = f"{datum['dialogue_id']}_{triple['head'].replace(' ','-')}_{triple['tail'].replace(' ', '-')}"
                        triple['instance_id'] = instance_id


TLiDB_path="TLiDB_Friends/TLiDB_Friends.json"

# Load original Friends data
friends_data = json.load(open(TLiDB_path, "r"))

generate_instance_ids(friends_data)

with open(TLiDB_path, "w") as f:
    json.dump(friends_data, f, indent=2)