import os
import json

data = json.load(open("original_clinc150.json","r"))
domains = json.load(open("clinc150_domains.json","r"))
domain_mapping = {v:key for key,values in domains.items() for v in values}

formatted_data = {
    "metadata":
    {
        "dataset_name":"clinc150",
        "tasks":[
            "intent_detection"
        ]
    },
    "data":[]
}

data_partitions = ["train","val","test"]
dialogue_id = 0
for p in data_partitions:
    for datum in data[p]:
        intent = datum[1]
        domain = domain_mapping[intent]

        formatted_datum = {
            "dialogue_id":f"{dialogue_id:05d}",
            "dialogue_metadata":{
                "intent_detection":None
            },
            "dialogue":[{
                "turn_id":0,
                "speaker":None,
                "utterance":datum[0],
                "intent_detection":{
                    "domain": domain,
                    "intent":intent
                }
            }]
        }
        formatted_data["data"].append(formatted_datum)
        dialogue_id += 1

TLiDB_path="TLiDB_clinc150"
if not os.path.isdir(TLiDB_path):
    os.mkdir(TLiDB_path)
TLiDB_path=os.path.join(TLiDB_path,"TLiDB_clinc150.json")
with open(TLiDB_path, "w") as f:
    json.dump(formatted_data, f, indent=2)