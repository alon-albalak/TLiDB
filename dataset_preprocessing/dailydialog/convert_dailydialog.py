import os
import json
from unidecode import unidecode


def format_data(original_data, formatted_data, partition):

    dialogue_id = 0
    for datum in original_data:
        formatted_datum = {
            "dialogue_id":"{}:{}".format(partition,dialogue_id),
            "dialogue_metadata":{
                "services":None,
                "dialogue_state_tracking":None,
                "dialogue_generation":None,
                "original_data_partition": partition
            },
            "dialogue":[]
        }
        for turn_id, turn in enumerate(datum.split('__eou__')):
            formatted_turn = {
                "turn_id":turn_id,
                "speakers":[turn_id%2],
                "utterance":unidecode(turn).strip(),
                "dialogue_state_tracking":{"domains":{}}
            }
            formatted_datum['dialogue'].append(formatted_turn)
        formatted_data['data'].append(formatted_datum)
    return

TLiDB_path="TLiDB_dailydialog"
if not os.path.isdir(TLiDB_path):
    os.mkdir(TLiDB_path)

formatted_data = {
    "metadata":
    {
        "dataset_name": "dailydialog",
        "tasks": [
            "dialogue_state_tracking",
        ],
        "task_metadata": {
            "dialogue_state_tracking": {"metrics": ["joint_slot_accuracy"]}
        }
    },
    "data": []
}

original_data_path="original_dailydialog/ijcnlp_dailydialog"
for root, dirs, files in os.walk(original_data_path):

    for file in files:
        if file in ["dialogues_{}.txt".format(x) for x in ["test","validation","train"]]:
            partition=root.split('/')[-1]
            f_path=os.path.join(root,file)
            with open(f_path, "r") as fin:
                original_data = fin.readlines()
            format_data(original_data, formatted_data, partition)


with open(f"{TLiDB_path}/TLiDB_dailydialog.json", "w") as f:
    json.dump(formatted_data,f,indent=2)
