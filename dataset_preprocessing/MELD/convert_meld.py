import pandas as pd
import json
import os

def format_data(in_file, formatted_data, fold):
    utterance_list = []
    prev_id = '0'
    for i, row in in_file.iterrows(): # Refactor This one
        dialog_id = fold + '-' + str(row['Dialogue_ID'])
        if str(row['Utterance_ID']) == '0':
            if prev_id != '0':
                formatted_datum = {"dialogue_id": prev_id,
                                   "dialogue_metadata": {"emotion_recognition": '', "original_data_partition": fold},
                                   "dialogue": utterance_list} # , "dialogue_level_task": ''}
                formatted_data['data'].append(formatted_datum)
                utterance_list = []
            prev_id = dialog_id

        utterance = {
            "turn_id": dialog_id + '-' + str(row['Utterance_ID']),
            'speakers':[row['Speaker']],
            "utterance":row['Utterance'],
            'emotion_recognition': row['Emotion'],
            # 'task_specific_metadata':''
        }
        utterance_list.append(utterance)

        if row['Emotion'] not in formatted_data['metadata']['task_metadata']['emotion_recognition']['labels']:
            formatted_data['metadata']['task_metadata']['emotion_recognition']['labels'].append(row['Emotion'])

    # last dialogue point
    formatted_datum = {"dialogue_id": dialog_id,
                       "dialogue_metadata": {"emotion_recognition": '', "original_data_partition": fold},
                       "dialogue": utterance_list}  # , "dialogue_level_task": ''}
    formatted_data['data'].append(formatted_datum)




TLiDB_path="TLiDB_MELD"
if not os.path.isdir(TLiDB_path):
    os.mkdir(TLiDB_path)

formatted_data = {
    "metadata":
    {
        "dataset_name": "MELD",
        "tasks": [
            "emotion_recognition"
        ],
        "task_metadata": {
            "emotion_recognition": {"labels": [], "metrics": ["f1", "accuracy"]}
        }
    },
    "data": []
}

data_partitions = [["train","train"],["dev","dev"],["test","test"]]
data_dir = "MELD.Raw/"
for p in data_partitions:
    data_path = data_dir + f"{p[0]}_sent_emo.csv"
    original_data = pd.read_csv(data_path, encoding='utf-8')
    format_data(original_data, formatted_data, p[1])


formatted_data['metadata']['task_metadata']['emotion_recognition']['labels'].sort()

with open(os.path.join(TLiDB_path,f"TLiDB_MELD.json"),"w") as f:
    json.dump(formatted_data, f, indent=2)




