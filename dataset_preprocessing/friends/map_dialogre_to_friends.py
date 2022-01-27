import json
from tqdm import tqdm
from difflib import SequenceMatcher
from utils import untokenize, remove_notes_from_utt

def get_dialogue_similarity(dialogre_datum, friends_datum):

    total_similarity = 0
    total_utterances = 0
    for i, d1 in enumerate(dialogre_datum):
        if len(d1['utterance']) < 10:
            continue
        best_similarity = 0
        for j, d2 in enumerate(friends_datum):
            if len(d2['utterance']) < 10:
                continue
            if i > j:
                continue
            if j > i + len(friends_datum) - len(dialogre_datum):
                continue
            s = SequenceMatcher(None, d1['utterance'], d2['utterance'])
            similarity = s.ratio()
            if similarity > best_similarity:
                best_similarity = similarity
        total_similarity += best_similarity
        total_utterances += 1
    dialogue_similarity = total_similarity / total_utterances
    return dialogue_similarity

def get_most_similar_dialogues(dialogre_dialogue, friends_without_notes):
    rankings = []
    for d in friends_without_notes:
        sim = get_dialogue_similarity(dialogre_dialogue, d['dialogue_without_notes'])
        rankings.append((sim, d['dialogue_id']))
    rankings = sorted(rankings, reverse=True)
    if rankings[0][0] > 0.9 and rankings[1][0] < 0.5:
        return [rankings[0]]
    else:
        return rankings[:5]

def parse_dialogre_dialogue(dialogre_dialogue):
    dialogue = []
    for line in dialogre_dialogue:
        speaker, utterance = line.split(":", 1)
        speaker = speaker.strip()
        utterance = utterance.strip()
        utterance = untokenize(utterance.split())
        dialogue.append({"speaker": speaker, "utterance": utterance})
    return dialogue

def remove_notes_and_note_utterances(friends_datum):
    new_dialogue = []
    for turn in friends_datum:
        new_utterance = ' '.join(remove_notes_from_utt(turn['utterance'].split()))
        if new_utterance:
            new_dialogue.append({"speaker": turn['speaker'], "utterance": new_utterance})
    return new_dialogue

def remove_notes_from_original_dialogues(friends_data):
    # add dialogues without notes to friends_data
    for datum in friends_data:
        datum['dialogue_without_notes'] = remove_notes_and_note_utterances(datum['dialogue'])
    return friends_data

def create_map(dialogre_data, friends_data):
    friends_data_without_notes = remove_notes_from_original_dialogues(friends_data['data'])

    for datum in tqdm(dialogre_data):
        dre_dialogue = parse_dialogre_dialogue(datum[0])
        similar_dialogues = get_most_similar_dialogues(dre_dialogue, friends_data_without_notes)
        datum.append(similar_dialogues)
    
    return dialogre_data

TLiDB_path="TLiDB_Friends/TLiDB_Friends.json"

# Load original Friends data
friends_data = json.load(open(TLiDB_path, "r"))

# load dialogRE data
data_partitions = ["train", "dev", "test"]
for p in data_partitions:
    dialogre_data = json.load(open(f"dialogre_{p}.json", "r"))
    # add annotations to original data
    dialogre_with_map = create_map(dialogre_data, friends_data)

    with open(f"dialogre_{p}_with_map.json", "w") as f:
        json.dump(dialogre_with_map, f, indent=2)
