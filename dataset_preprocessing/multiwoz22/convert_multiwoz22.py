import os
import json


def format_data(original_data, formatted_data, partition):

    for datum in original_data:
        formatted_datum = {
            "dialogue_id":datum['dialogue_id'],
            "dialogue_metadata":{
                "services":datum['services'],
                "dialogue_state_tracking":None,
                "dialogue_generation":None,
                "original_data_partition": partition
            },
            "dialogue":[]
        }
        for turn in datum['turns']:
            formatted_turn = {
                "turn_id":turn['turn_id'],
                "speakers":[turn['speaker']],
                "utterance":turn['utterance'],
                "dialogue_state_tracking":{"domains":{}}
            }

            slot_keys = ['exclusive_end','slot','start','value','copy_from','copy_from_value']
            for domain in turn['frames']:
                formatted_domain = {}
                for slot in domain['slots']:
                    for k in slot_keys:
                        if k not in slot:
                            slot[k] = []
                formatted_domain['slots'] = domain['slots']
                if turn['speaker'] == "USER":
                    formatted_domain['state'] = domain['state']
                formatted_turn['dialogue_state_tracking']['domains'][domain['service']] = formatted_domain
            formatted_datum['dialogue'].append(formatted_turn)
        formatted_data['data'].append(formatted_datum)
    return

TLiDB_path="TLiDB_multiwoz22"
if not os.path.isdir(TLiDB_path):
    os.mkdir(TLiDB_path)

formatted_data = {
    "metadata":
    {
        "dataset_name": "multiwoz22",
        "tasks": [
            "dialogue_state_tracking",
        ],
        "task_metadata": {
            "dialogue_state_tracking": {"metrics": ["joint_slot_accuracy"]}
        }
    },
    "data": []
}

original_data_path="original_multiwoz22"
for root, dirs, files in os.walk(original_data_path):

    for file in files:
        if file.endswith(".json"):
            partition=root.split('/')[-1]
            f_path=os.path.join(root,file)
            original_data = json.load(open(f_path, "r"))
            format_data(original_data, formatted_data, partition)


with open(f"{TLiDB_path}/TLiDB_multiwoz22.json", "w") as f:
    json.dump(formatted_data,f,indent=2)
