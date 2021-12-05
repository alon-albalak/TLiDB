import os
import json
from utils import untokenize

emo_dict = {'0': 'neutral', '1': 'anger', '2': 'disgust', '3': 'fear', '4': 'happiness', '5': 'sadness', '6': 'surprise'}
da_dict = {'1': 'inform', '2': 'question', '3': 'directive', '4': 'commissive'}
topic_dict = {'1': 'Ordinary Life', '2': 'School Life', '3': 'Culture & Education',
                '4': 'Attitude & Emotion', '5': 'Relationship', '6': 'Tourism' ,
                '7': 'Health', '8': 'Work', '9': 'Politics', '10': 'Finance'}
dialogue_id_counter = 1

def format_data(original_data, formatted_data, partition):
    """Updates formatted_data inplace with the data from original_data"""
    global dialogue_id_counter
    data_file = original_data['text']
    emotion_label_file = original_data['emotion_label']
    dialogue_act_label_file = original_data['dialogue_act_label']
    topic_label_file = original_data['topic_label']

    for raw_dialogue, emo_label, da_label, topic_label in  zip(data_file, emotion_label_file, dialogue_act_label_file, topic_label_file):
        dialogue = [untokenize(u.split()) for u in raw_dialogue.strip().split('__eou__')[:-1]]
        emo_labels = [emo_dict[l] for l in emo_label.strip().split(' ')]
        da_labels = [da_dict[l] for l in da_label.strip().split(' ')]
        topic_label = topic_dict[topic_label.strip()]

        dialogue_id = 'dialogue-' + str(dialogue_id_counter)
        dialogue_id_counter += 1
        formatted_datum = {
            "dialogue_id": dialogue_id,
            "topic_classification":topic_label,
            "dialogue_metadata": {
                "emotion_recognition": None,
                "dialogue_act_classification": None,
                "topic_classification": None,
                "original_data_partition": partition
            },
            "dialogue": []
        }

        speaker_id=1
        for i, (ut, emo, da) in enumerate(zip(dialogue, emo_labels, da_labels)):
            formatted_turn = {
                "turn_id": str(i+1),
                "speakers": [f"speaker{speaker_id}"],
                "utterance": ut,
                "emotion_recognition": emo,
                "dialogue_act_classification": da
            }
            formatted_datum['dialogue'].append(formatted_turn)
            if speaker_id == 1:
                speaker_id = 2
            else:
                speaker_id = 1
        formatted_data['data'].append(formatted_datum)
    return

TLiDB_path="TLiDB_DailyDialog"
if not os.path.isdir(TLiDB_path):
    os.mkdir(TLiDB_path)

formatted_data = {
    "metadata":
    {
        "dataset_name": "DailyDialog",
        "tasks": [
            "emotion_recognition",
            "dialogue_act_classification",
            "topic_classification"
        ],
        "task_metadata": {
            "emotion_recognition": {"labels": list(emo_dict.values()), "metrics": ["f1", "accuracy"]},
            "dialogue_act_classification": {"labels": list(da_dict.values()), "metrics": ["f1", "accuracy"]},
            "topic_classification": {"labels":list(topic_dict.values()), "metrics": ["f1", "accuracy"]}
        }
    },
    "data": []
}

data_partitions = [["train","train"],["validation","dev"],["test","test"]]

for p in data_partitions:
    data_path = f"{p[0]}"
    original_data = {'text':open(data_path + '/dialogues_' +  p[0] + '.txt', "r"),
                     'emotion_label':open(data_path + '/dialogues_emotion_' +  p[0] + '.txt', "r"),
                     'dialogue_act_label':open(data_path + '/dialogues_act_' +  p[0] + '.txt', "r"),
                     'topic_label':open(data_path + '/dialogues_topic_' +  p[0] + '.txt', "r")
                     }
    format_data(original_data, formatted_data, p[1])

formatted_data['metadata']['task_metadata']['emotion_recognition']['labels'].sort()

with open(os.path.join(TLiDB_path,f"TLiDB_DailyDialog.json"),"w", encoding='utf8') as f:
    json.dump(formatted_data, f, indent=2, ensure_ascii=False)
