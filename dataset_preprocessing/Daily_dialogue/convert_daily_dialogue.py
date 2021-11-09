import os
import json
# import zip

emo_dict = {'0': 'no emotion', '1': 'anger', '2': 'disgust', '3': 'fear', '4': 'happiness', '5': 'sadness', '6': 'surprise'}

def format_data(original_data, formatted_data, partition):
    """Updates formatted_data inplace with the data from original_data"""
    print(partition)
    dialogue_id_counter = 1
    data_file = original_data['text']
    label_file = original_data['label']

    for raw_dialogue, label in  zip(data_file, label_file):
        dialogue = raw_dialogue.strip().split('__eou__')[:-1]
        labels = label.strip().split(' ')

        dialogue_id = partition + '-' + str(dialogue_id_counter)
        dialogue_id_counter += 1
        formatted_datum = {
            "dialogue_id": dialogue_id,
            "dialogue_metadata": {
                "emotion_recognition": None,
                # "dialogue_actions": None,
                "original_data_partition": partition
            },
            "dialogue": []
        }

        for i, ut in enumerate(dialogue):
            if emo_dict[labels[i]] not in formatted_data['metadata']['task_metadata']['emotion_recognition']['labels']:
                formatted_data['metadata']['task_metadata']['emotion_recognition']['labels'].append(labels[i])

            formatted_turn = {
                "turn_id": dialogue_id + '-' + str(i),
                "speakers": '', # ut['speakers'], # TODO: check this one later... no specific speaker is mentioned
                "utterance": ut,
                "emotion_recognition": labels[i],
                # "dialogue_actions": ''
            }
            formatted_datum['dialogue'].append(formatted_turn)
        formatted_data['data'].append(formatted_datum)
    return

TLiDB_path="TLiDB_Daily_dialogue"
if not os.path.isdir(TLiDB_path):
    os.mkdir(TLiDB_path)

formatted_data = {
    "metadata":
    {
        "dataset_name": "daily_dialogue",
        "tasks": [
            "emotion_recognition",
            "dialogue_actions"
        ],
        "task_metadata": {
            "emotion_recognition": {"labels": [], "metrics": ["f1", "accuracy"]},
            # "dialogue_actions": {'labels':[], "metrics":["f1", "accuracy"]}
        }
    },
    "data": []
}

data_partitions = [["train","train"],["validation","dev"],["test","test"]]

for p in data_partitions:
    data_path = f"{p[0]}"
    # original_data = json.load(open(data_path,"r"))
    original_data = {'text':open(data_path + '/dialogues_' +  p[0] + '.txt', "r"),
                     'label':open(data_path + '/dialogues_emotion_' +  p[0] + '.txt', "r")}
    format_data(original_data, formatted_data, p[1])

formatted_data['metadata']['task_metadata']['emotion_recognition']['labels'].sort()

with open(os.path.join(TLiDB_path,f"TLiDB_Daily_Dialogue.json"),"w") as f:
    json.dump(formatted_data, f, indent=2)
