import os
import json


def format_data(data):
    formatted_data = {
        "metadata":
        {
            "dataset_name":"multiwoz22",
            "tasks":[
                "dialogue_state_tracking",
            ],
            "task_metadata":{
                "dialogue_state_tracking":{"metrics":["joint_slot_accuracy"]}
            }
        },
        "data":[]
    }

    for datum in data:
        formatted_datum = {
            "dialogue_id":datum['dialogue_id'],
            "dialogue_metadata":{
                "services":datum['services'],
                "dialogue_state_tracking":None,
                "dialogue_generation":None
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
    return formatted_data

TLiDB_path="TLiDB_multiwoz22"
if not os.path.isdir(TLiDB_path):
    os.mkdir(TLiDB_path)

original_data_path="original_multiwoz22"
for root, dirs, files in os.walk(original_data_path):

    for file in files:
        if file.endswith(".json"):

            f_path=os.path.join(root,file)
            data = json.load(open(f_path, "r"))
            formatted_data = format_data(data)

            with open(f"{TLiDB_path}/{root.split('/')[-1]}_{file}", "w") as f:
                json.dump(formatted_data,f,indent=2)
