import csv
import os
import json

data_partitions = ["train", "test"]
data = {p: list(csv.reader(open(f"{p}.csv", "r"), delimiter=','))[1:] for p in data_partitions}
intents = json.load(open("categories.json", "r"))

formatted_data = {
    "metadata":
    {
        "dataset_name": "banking77",
        "tasks": [
            "intent_detection"
        ],
        "task_metadata":
        {"intent_detection": {"labels": intents, "metrics": ["f1","accuracy"]}}
    },
    "data": []
}
domain = "banking"

dialogue_id = 0
for p in data_partitions:
    for utterance, intent in data[p]:
        assert intent in intents

        formatted_datum = {
            "dialogue_id": f"{dialogue_id:05d}",
            "dialogue_metadata": {
                "intent_detection": None,
                "original_data_partition": p
            },
            "dialogue":[{
                "turn_id": 0,
                "speakers": [],
                "utterance": utterance,
                "intent_detection":{
                    "domain": domain,
                    "intent": intent
                }
            }]
        }
        formatted_data["data"].append(formatted_datum)
        dialogue_id += 1


TLiDB_path = "TLiDB_banking77"
os.makedirs(TLiDB_path, exist_ok=True)

TLiDB_path = os.path.join(TLiDB_path, "TLiDB_banking77.json")
with open(TLiDB_path, "w") as f:
    json.dump(formatted_data, f, indent=2)
