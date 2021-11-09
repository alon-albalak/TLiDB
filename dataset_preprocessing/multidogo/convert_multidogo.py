import csv
import os
import json

domains = ["airline", "fastfood", "finance", "insurance", "media", "software"]
data_partitions = ["train", "dev", "test"]

intents = []

formatted_data = {
    "metadata":
    {
        "dataset_name": "multidogo",
        "tasks": [
            "intent_detection", "dialog_state_tracking"
        ],
        "task_metadata": {
            "intent_detection": {"labels": intents, "metrics": ["f1","accuracy"]},
            "dialogue_state_tracking": {"metrics": ["joint_slot_accuracy"]}
        }
    },
    "data": []
}

for domain in domains:
    for p in data_partitions:
        data = list(csv.reader(open(f"{domain}_{p}.tsv", "r"), delimiter='\t'))[1:]

        for dialogue_id, turn_id, _, utterance, slots, intent in data:
            formatted_datum = {
                "dialogue_id": dialogue_id,
                "dialogue_metadata": {
                    "intent_detection": None,
                    "original_data_partition": p
                },
                "dialogue": [{
                    "turn_id": turn_id,
                    "speakers": [],
                    "utterance": utterance,
                    "intent_detection": {
                        "domain": domain,
                        "intent": intent
                    },
                    "dialogue_state_tracking": {
                        "domain": domain,
                        "slots": slots
                    }
                }]
            }
            formatted_data["data"].append(formatted_datum)


TLiDB_path = "TLiDB_multidogo"
os.makedirs(TLiDB_path, exist_ok=True)

TLiDB_path = os.path.join(TLiDB_path, "TLiDB_multidogo.json")
with open(TLiDB_path, "w") as f:
    json.dump(formatted_data, f, indent=2)
