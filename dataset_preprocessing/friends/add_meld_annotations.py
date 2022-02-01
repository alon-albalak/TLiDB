import json
import csv
from tqdm import tqdm
from meld_manual_mapping import manual_mapping

MELD_EMOTION_fields = ["serial_number", "utterance", "speaker","emotion","sentiment","dialogue_id","utterance_id","season","episode","start_time","end_time"]

def get_friends_datum_from_id(friends_data, friends_id):
    for d in friends_data['data']:
        if d['dialogue_id'] == friends_id:
            return d
    raise ValueError(f"No friends datum with dialogue_id {friends_id}")

def get_friends_turn_from_id(friends_datum, turn_id):
    for turn in friends_datum['dialogue']:
        if turn['turn_id'] == int(turn_id):
            return turn
    raise ValueError(f"No friends turn with turn_id {turn_id}")

def add_meld_annotations(meld_data, friends_data, partition):
    if "MELD_emotion_recognition" not in friends_data["metadata"]["tasks"]:
        friends_data['metadata']['tasks'].append("MELD_emotion_recognition")
        friends_data['metadata']['task_metadata']['MELD_emotion_recognition'] = {
            "labels": ["anger","disgust","fear","joy","neutral","sadness","surprise"],
            "metrics": ["f1"],
            "metric_kwargs": {"f1": [{"average": "macro"}, {"average": "micro"}]}
        }

    label_distribution = {label: 0 for label in friends_data['metadata']['task_metadata']['MELD_emotion_recognition']['labels']}

    tot, multilabeled, num_emotion_overlap = 0, 0, 0

    for datum in tqdm(meld_data):
        meld_ID = f"{partition}-{datum[MELD_EMOTION_fields.index('serial_number')]}"

        for friends_utt_id in manual_mapping[meld_ID]:
            friends_dialogue_id, utt_id = friends_utt_id.rsplit("_u", 1)
            d = get_friends_datum_from_id(friends_data, friends_dialogue_id)
            turn = get_friends_turn_from_id(d, utt_id)

            if "MELD_emotion_recognition" not in d['dialogue_metadata']:
                d['dialogue_metadata']['MELD_emotion_recognition'] = None

            # MELD datums do not split utterances the same as friends data, so some friends datum correspond to multiple annotations from MELD
            #   we allow emotions to be overwritten so that the annotations for the end of the dialogue is selected, unless it is neutral
            if 'MELD_emotion_recognition' in turn:
                multilabeled += 1
                tot -= 1
                if datum[MELD_EMOTION_fields.index("emotion")] != "neutral":
                    label_distribution[datum[MELD_EMOTION_fields.index("emotion")]] += 1
                    label_distribution[turn['MELD_emotion_recognition']] -= 1
                    turn['MELD_emotion_recognition'] = datum[MELD_EMOTION_fields.index("emotion")]

            else:
                turn['MELD_emotion_recognition'] = datum[MELD_EMOTION_fields.index("emotion")]
                label_distribution[turn['MELD_emotion_recognition']] += 1

            if 'emotion_recognition' in turn:
                num_emotion_overlap += 1
            tot += 1

    print(f"Total samples from MELD: {tot}")
    print(f"Multilabeled samples in MELD: {multilabeled}")
    print(f"MELD samples overlapping with emoryNLP emotion: {num_emotion_overlap}")
    print(f"Label distribution: {label_distribution}")

TLiDB_path="TLiDB_Friends/TLiDB_Friends.json"

# Load original Friends data
friends_data = json.load(open(TLiDB_path, "r"))


data_partitions = ["train", "dev", "test"]
for p in data_partitions:
    meld_data = list(csv.reader(open(f"meld_{p}.csv", "r")))[1:]
    add_meld_annotations(meld_data, friends_data, p)

with open(TLiDB_path, "w") as f:
    json.dump(friends_data, f, indent=2)