import os
import json

def format_data(original_data, formatted_data, partition):
    """Updates formatted_data inplace with the data from original_data"""
    print(partition)
    for ep in original_data['episodes']:
        for sc in ep['scenes']:
            formatted_datum = {
                "dialogue_id":sc['scene_id'],
                "dialogue_metadata":{
                    "emotion_recognition":None,
                    "original_data_partition":partition
                },
                "dialogue":[]
            }
            for ut in sc['utterances']:
                if ut['emotion'] not in formatted_data['metadata']['task_metadata']['emotion_recognition']['labels']:
                    formatted_data['metadata']['task_metadata']['emotion_recognition']['labels'].append(ut['emotion'])

                formatted_turn = {
                    "turn_id":ut['utterance_id'],
                    "speakers":ut['speakers'],
                    "utterance":ut['transcript'],
                    "emotion_recognition":ut['emotion']
                }
                formatted_datum['dialogue'].append(formatted_turn)
            formatted_data['data'].append(formatted_datum)
    return

TLiDB_path="TLiDB_friends_ER"
if not os.path.isdir(TLiDB_path):
    os.mkdir(TLiDB_path)

formatted_data = {
    "metadata":
    {
        "dataset_name": "friends_ER",
        "tasks": [
            "emotion_recognition"
        ],
        "task_metadata": {
            "emotion_recognition": {"labels": [], "metrics": ["f1", "accuracy"]}
        }
    },
    "data": []
}

data_partitions = [["trn","train"],["dev","dev"],["tst","test"]]

for p in data_partitions:
    data_path = f"emotion-detection-{p[0]}.json"
    original_data = json.load(open(data_path,"r"))
    format_data(original_data, formatted_data, p[1])

formatted_data['metadata']['task_metadata']['emotion_recognition']['labels'].sort()

with open(os.path.join(TLiDB_path,f"TLiDB_friends_ER.json"),"w") as f:
    json.dump(formatted_data, f, indent=2)
