import json
from tqdm import tqdm
from difflib import SequenceMatcher
from utils import untokenize, remove_notes_from_utt
from dialogre_manual_mapping import manual_mapping, unmappable

relation_map = {
    1: "per:positive_impression",
    2: "per:negative_impression",
    3: "per:acquaintance",
    4: "per:alumni",
    5: "per:boss",
    6: "per:subordinate",
    7: "per:client",
    8: "per:dates",
    9: "per:friends",
    10: "per:girl/boyfriend",
    11: "per:neighbor",
    12: "per:roommate",
    13: "per:children",
    14: "per:other_family",
    15: "per:parents",
    16: "per:siblings",
    17: "per:spouse",
    18: "per:place_of_residence",
    19: "per:place_of_birth",  # does not exist in training set
    20: "per:visited_place",
    21: "per:origin",
    22: "per:employee_or_member_of",
    23: "per:schools_attended",
    24: "per:works",
    25: "per:age",
    26: "per:date_of_birth",
    27: "per:major",
    28: "per:place_of_work",
    29: "per:title",
    30: "per:alternate_names",
    31: "per:pet",
    32: "gpe:residents_of_place",
    34: "gpe:visitors_of_place",
    33: "gpe:births_in_place",  # does not exist in training set
    35: "org:employees_or_members",
    36: "org:students",
    37: "unanswerable",
}

def remove_notes_and_note_utterances(friends_datum):
    new_dialogue = []
    for turn in friends_datum:
        new_utterance = ' '.join(remove_notes_from_utt(turn['utterance'].split()))
        if new_utterance:
            new_dialogue.append({"turn_id": turn['turn_id'], "speakers": turn['speakers'], "utterance": new_utterance})
    return new_dialogue

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

def parse_dialogre_dialogue(dialogre_dialogue):
    dialogue = []
    for line in dialogre_dialogue:
        speaker, utterance = line.split(":", 1)
        speaker = speaker.strip()
        utterance = utterance.strip()
        utterance = untokenize(utterance.split())
        dialogue.append({"speakers": speaker, "utterance": utterance})
    return dialogue

def map_entity_to_dialogue(entity, speaker_map):
    if 'Speaker' in entity:
        if entity in speaker_map:
            return speaker_map[entity]
    return entity

def get_friends_datum_from_dialogue_id(friends_data, dialogue_id):
    for d in friends_data:
        if d['dialogue_id'] == dialogue_id:
            return d

def get_relevant_turns_with_speaker_map(friends_datum, dialogre_datum):
    relevant_turns = []
    speaker_map = {}
    found_dre = 0

    original_turn, dre_turn = 0, 0
    for i in range(len(friends_datum)):
        dd = untokenize(dialogre_datum[dre_turn]['utterance'].split())
        od = friends_datum[original_turn]['utterance']

        s = SequenceMatcher(None, dd, od)
        similarity = s.ratio()

        if dd == od:
            relevant_turns.append(friends_datum[original_turn]['turn_id'])

            # map speakers
            if len(friends_datum[original_turn]['speakers']) == 1:
                speaker_map[dialogre_datum[dre_turn]['speakers']] = friends_datum[original_turn]['speakers'][0]
            else:
                speakers = dialogre_datum[dre_turn]['speakers'].split(",")
                if len(speakers) == len(friends_datum[original_turn]['speakers']):
                    for speaker_num, speaker in zip(speakers, friends_datum[original_turn]['speakers']):
                        speaker_num = speaker_num.strip()
                        if speaker_num not in speaker_map:
                            speaker_map[speaker_num] = speaker

            dre_turn += 1
            found_dre += 1

        elif min(len(dd),len(od)) > 10 and similarity > 0.8:
            relevant_turns.append(friends_datum[original_turn]['turn_id'])

            if len(friends_datum[original_turn]['speakers']) == 1:
                speaker_map[dialogre_datum[dre_turn]['speakers']] = friends_datum[original_turn]['speakers'][0]
            else:
                speakers = dialogre_datum[dre_turn]['speakers'].split(",")
                if len(speakers) == len(friends_datum[original_turn]['speakers']):
                    for speaker_num, speaker in zip(speakers, friends_datum[original_turn]['speakers']):
                        speaker_num = speaker_num.strip()
                        if speaker_num not in speaker_map:
                            speaker_map[speaker_num] = speaker

            dre_turn += 1
            found_dre += 1

        elif min(len(od), len(dd)) <= 10 and abs(len(od)-len(dd)) < 3:
            
            dd = dd.replace(" ","")
            od = od.replace(" ","")
            s = SequenceMatcher(None, dd, od)
            similarity = s.ratio()

            if similarity > 0.6:
                relevant_turns.append(friends_datum[original_turn]['turn_id'])
                dre_turn += 1
                found_dre += 1
        original_turn += 1

        if dre_turn >= len(dialogre_datum):
            break

    if found_dre < len(dialogre_datum):
        # if the last turn is the missing one, handle the case differently
        if len(dialogre_datum) - found_dre == 1:
            assert(relevant_turns)
            for turn in friends_datum:
                if turn['turn_id'] == relevant_turns[-1]+1:
                    last_friends_turn = turn
                    break
            s = SequenceMatcher(None, untokenize(dialogre_datum[dre_turn]['utterance'].split()), last_friends_turn['utterance'])
            similarity = s.ratio()
            if similarity > 0.5:
                relevant_turns.append(last_friends_turn['turn_id'])
                if len(last_friends_turn['speakers']) == 1:
                    speaker_map[dialogre_datum[dre_turn]['speakers']] = last_friends_turn['speakers'][0]
                else:
                    speakers = dialogre_datum[dre_turn]['speakers'].split(",")
                    if len(speakers) == len(friends_datum[original_turn]['speakers']):
                        for speaker_num, speaker in zip(speakers, friends_datum[original_turn]['speakers']):
                            speaker_num = speaker_num.strip()
                            if speaker_num not in speaker_map:
                                speaker_map[speaker_num] = speaker
                  
                dre_turn += 1
                found_dre += 1

        while found_dre < len(dialogre_datum):
            if relevant_turns:
                missing_turn = relevant_turns[-1] + 1
            else:
                missing_turn = 0
            second_half_turns, second_half_speakers = get_relevant_turns_with_speaker_map(friends_datum[missing_turn:], dialogre_datum[found_dre+1:])

            speaker_map.update(second_half_speakers)

            if missing_turn == 0:
                missing_turn = second_half_turns[0]-1
                
            relevant_turns += [missing_turn] + second_half_turns
            found_dre = len(relevant_turns)

    # if the first turn starts from just after the scene info
    if relevant_turns[0] == 1:
        relevant_turns.insert(0, 0)

    return relevant_turns, speaker_map

def add_dialogre_annotations(dialogre_data, friends_data, partition):
    if "relation_extraction" not in friends_data['metadata']['tasks']:
        friends_data['metadata']['tasks'].append("relation_extraction")
        friends_data['metadata']['task_metadata']['relation_extraction'] = {
            'labels': list(relation_map.values()),
            'metrics': ['multilabel_f1','mean_reciprocal_rank'],
            "metric_kwargs": {
                "multilabel_f1": [{'labels':[i for i in range(36) if i != 29]}], # dont include "unanswerable" or "per:alternate_names"
                "mean_reciprocal_rank": [{'labels':[i for i in range(36) if i != 29]}]} # dont include "unanswerable" or "per:alternate_names"
        }
    triple_counts = {v: 0 for k, v in relation_map.items()}

    # track dialogues which fail
    #   some dialogues in dialogRE cross multiple scenes from emoryNLP data 
    #   some dialogues simply don't exist in emoryNLP data
    failed = 0
    for i, datum in enumerate(tqdm(dialogre_data)):
        dialogre_id = f"{partition}_{i}"

        # some dialogre datums cover multiple dialogues, and some don't exist in the emoryNLP dataset
        if dialogre_id in unmappable:
            failed += 1
            continue

        dre_dialogue = parse_dialogre_dialogue(datum[0])
        possible_matches = datum[2]

        if dialogre_id in manual_mapping:
            dialogue_id = manual_mapping[dialogre_id]['dialogue_id']
            start_turn,end_turn = manual_mapping[dialogre_id]['turns']
            speaker_map = manual_mapping[dialogre_id]['speaker_map']
            d = get_friends_datum_from_dialogue_id(friends_data['data'], dialogue_id)
            dialogue_without_notes = remove_notes_and_note_utterances(d['dialogue'])
            turns = [j for j, utt in enumerate(d['dialogue']) if start_turn<=j<=end_turn]

        elif possible_matches[0][0] > 0.9 and all([p[0] < 0.7 for p in possible_matches[1:]]):
            # we found a definite match
            dialogue_id = datum[2][0][1]
            # first, get the datum without notes in order to compare 
            d = get_friends_datum_from_dialogue_id(friends_data['data'], dialogue_id)
            dialogue_without_notes = remove_notes_and_note_utterances(d['dialogue'])
            try:
                turns, speaker_map = get_relevant_turns_with_speaker_map(dialogue_without_notes, dre_dialogue)
            except:
                failed += 1
        else:
            raise Exception(f"Could not find a match for {dialogre_id}")



        dialogre_annotation = {
            'turns': turns,
            "relation_triples": []
        }

        for entity_pair in datum[1]:
            relations = []
            head = map_entity_to_dialogue(entity_pair['x'], speaker_map)
            tail = map_entity_to_dialogue(entity_pair['y'], speaker_map)
            for r in entity_pair['r']:
                relations.append(r)
                triple_counts[r] += 1
            
            # alternate names is not a valid relation for our data format
            if relations != ['per:alternate_names']:

                dialogre_annotation['relation_triples'].append({
                    'head': head,
                    'tail': tail,
                    'relations': relations,
                })

        # only add annotations when they have valid relations
        if dialogre_annotation['relation_triples']:
            if 'relation_extraction' not in d.keys():
                d['relation_extraction'] = []
                d['dialogue_metadata']['relation_extraction'] = None
            d['relation_extraction'].append(dialogre_annotation)

    print(f"Failed to match {failed} dialogre datums")
    print(f"{triple_counts}")
    return friends_data

TLiDB_path="TLiDB_Friends/TLiDB_Friends.json"

# Load original Friends data
friends_data = json.load(open(TLiDB_path, "r"))

# load dialogRE data
data_partitions = ["train", "dev", "test"]
for p in data_partitions:
    dialogre_data = json.load(open(f"dialogre_{p}_with_map.json", "r"))
    # add annotations to original data
    friends_data = add_dialogre_annotations(dialogre_data, friends_data, p)

with open(TLiDB_path, "w") as f:
    json.dump(friends_data, f, indent=2)