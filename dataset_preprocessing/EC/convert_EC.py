import os
import json
import pandas as pd

def format_data(original_data, formatted_data, partition):
    """Updates formatted_data inplace with the data from original_data"""
    for i, row in original_data.iterrows():
        dialog_id = partition + '-' + str(row['id'])
        formatted_datum = {
            "dialogue_id": dialog_id,
            "dialogue_metadata": {
                "emotion_recognition": True,
                "original_data_partition": partition
            },
            "dialogue": []
        }
        for j in range(3):
            formatted_turn = {
                "turn_id": dialog_id + '-' + str(j+1),
                "speakers": '',   # no specific speaker is mentioned
                "utterance": row['turn' + str(j+1)],
                "emotion_recognition": row['label'],
            }
            formatted_datum['dialogue'].append(formatted_turn)

        if row['label'] not in formatted_data['metadata']['task_metadata']['emotion_recognition']['labels']:
            formatted_data['metadata']['task_metadata']['emotion_recognition']['labels'].append(row['label'])
        formatted_data['data'].append(formatted_datum)
    return

TLiDB_path="TLiDB_EC"
if not os.path.isdir(TLiDB_path):
    os.mkdir(TLiDB_path)

formatted_data = {
    "metadata":
    {
        "dataset_name": "Emotion_Context",
        "tasks": [
            "emotion_recognition",
            "dialogue_actions"
        ],
        "task_metadata": {
            "emotion_recognition": {"labels": [], "metrics": ["f1", "accuracy"]},
        }
    },
    "data": []
}

data_partitions = [["starterkitdata/train","train"],["test/test","test"]]

for p in data_partitions:
    data_path = "ec_data"
    original_data = pd.read_csv(data_path + '/' + p[0] + '.txt', sep='\t')
    format_data(original_data, formatted_data, p[1])

formatted_data['metadata']['task_metadata']['emotion_recognition']['labels'].sort()

with open(os.path.join(TLiDB_path,f"TLiDB_EC.json"),"w") as f:
    json.dump(formatted_data, f, indent=2)
