import os
import json

def format_data(data):
    formatted_data = {
        "metadata":
        {
            "dataset_name":"friends_ER",
            "tasks":[
                "emotion_recognition"
            ],
            "task_metadata":{
                "emotion_recognition":{"labels":[],"metrics":["f1","accuracy"]}
            }
        },
        "data":[]
    }

    for ep in data['episodes']:
        for sc in ep['scenes']:
            formatted_datum = {
                "dialogue_id":sc['scene_id'],
                "dialogue_metadata":{
                    "emotion_recognition":None
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
    formatted_data['metadata']['task_metadata']['emotion_recognition']['labels'].sort()
    return formatted_data

TLiDB_path="TLiDB_friends_ER"
if not os.path.isdir(TLiDB_path):
    os.mkdir(TLiDB_path)
data_partitions = [["trn","train"],["dev","dev"],["tst","test"]]
for p in data_partitions:
    data_path = f"emotion-detection-{p[0]}.json"
    data = json.load(open(data_path,"r"))
    formatted_data = format_data(data)
    with open(os.path.join(TLiDB_path,f"emotion-detection-{p[1]}.json"),"w") as f:
        json.dump(formatted_data, f, indent=2)