import json
import csv
from tqdm import tqdm
import re
from difflib import SequenceMatcher
from utils import untokenize, remove_notes_from_utt

def parse_pd_dialogue(persona_data_dialogue):
    dialogue = []
    for line in persona_data_dialogue.split("<br><br>")[1:-1]:
        speaker = re.search(f'<b>(.*)</b>', line)
        if speaker:
            speaker = speaker.group(1)
            utterance = re.search(f'</b>: (.*)', line).group(1)
        else:
            assert("<b>" not in line)
            assert("</b>" not in line)
            assert("<br>" not in line)
            utterance = line
            speaker = ""
        dialogue.append({"speaker": speaker, "utterance": utterance})
    return dialogue

def get_partial_dialogue(original_dialogue, pd_dialogue):
    turns = []

    found_pd = 0

    original_turn, pd_turn = 0, 0
    for i in range(len(original_dialogue)):
        pd = ' '.join(remove_notes_from_utt(untokenize(pd_dialogue[pd_turn]['utterance'].split()).split()))
        od = ' '.join(remove_notes_from_utt(original_dialogue[original_turn]['utterance'].split()))
        
        s = SequenceMatcher(None, pd, od)
        similarity = s.ratio()

        if od == pd:
            turns.append(original_dialogue[original_turn]['turn_id'])
            pd_turn += 1
            found_pd += 1
        elif min(len(od),len(pd)) > 10 and similarity > 0.8:
            turns.append(original_dialogue[original_turn]['turn_id'])
            pd_turn += 1
            found_pd += 1
        elif len(od) < 10 and len(pd) < 10 and similarity > 0.7:
            turns.append(original_dialogue[original_turn]['turn_id'])
            pd_turn += 1
            found_pd += 1
        original_turn += 1
        if pd_turn >= len(pd_dialogue):
            break

    if found_pd < len(pd_dialogue):

        while found_pd < len(pd_dialogue):
            missing_turn = turns[-1]+1
            second_half_turns = get_partial_dialogue(original_dialogue[missing_turn+1:], pd_dialogue[found_pd+1:])
            
            turns += [missing_turn] + second_half_turns
            found_pd = len(turns)

    return turns

def same_dialogue(d1, d2):
    if len(d1) != len(d2):
        return False
    for i in range(len(d1)):
        if d1[i]['speaker'] != d2[i]['speaker'] or d1[i]['utterance'] != d2[i]['utterance']:
            return False
    return True

def get_friends_datum_from_pd_data(friends_data, scene_id, pd_dialogue):
    for i, d in enumerate(friends_data['data']):
        if d['dialogue_id'] == 's'+scene_id:
            if same_dialogue(d['dialogue'], pd_dialogue):
                return d, [i for i in range(len(d['dialogue']))]
            else:
                turns = get_partial_dialogue(d['dialogue'], pd_dialogue)
                if turns:
                    assert turns == [i for i in range(turns[0], turns[-1]+1)], "PD Dialogue skips some original turns"
                    return d, turns
    return None, None

def add_personality_detection_annotations(friends_data, personality_detection_data, headers):
    if "personality_detection" not in friends_data['metadata']['tasks']:
        friends_data['metadata']['tasks'].append("personality_detection")
        friends_data['metadata']['task_metadata']['personality_detection'] = {"metrics": ["accuracy"]}

    SCENE_ID = headers.index('scene_id')
    TEXT = headers.index('text')
    FOCUS_SPEAKER = headers.index('character')
    AGR = headers.index("cAGR")
    CON = headers.index("cCON")
    EXT = headers.index("cEXT")
    OPN = headers.index("cOPN")
    NEU = headers.index("cNEU")

    # for scenes with dialogues that are too complex for automatic matching,
    #       simply list them here with their matching dialogue and turns
    scene_exceptions = {
        114:{"data_index":228,"turns":[33,50]},
        141:{"data_index":260,"turns":[6,16]},
        301:{"data_index":563,"turns":[7,19]},
        387:{"data_index":724,"turns":[132,146]},
        389:{"data_index":724,"turns":[74,83]},
        438:{"data_index":835,"turns":[9,17]},
        456:{'data_index':886,'turns':[13,27]},
        458:{'data_index':886,'turns':[24,37]},
        491:{'data_index':958,'turns':[77,83]},
        566:{'data_index':1084,'turns':[14,20]},
        704:{'data_index':1295,'turns':[1,9]},
        }

    for i, persona_data in enumerate(tqdm(personality_detection_data)):

        pd_dialogue = parse_pd_dialogue(persona_data[TEXT])

        # fix specific utterances with typos
        if persona_data[SCENE_ID] == '04_e03_c07':
            pd_dialogue[3]['utterance'] += ')'
        if i==685:
            pd_dialogue[6]['utterance'] += ")"

        if i in scene_exceptions:
            d = friends_data['data'][scene_exceptions[i]['data_index']]
            turns = [j for j in range(scene_exceptions[i]['turns'][0], scene_exceptions[i]['turns'][1]+1)]
        else:
            d, turns = get_friends_datum_from_pd_data(friends_data, persona_data[SCENE_ID], pd_dialogue)


        pd_annotation = {}
        pd_annotation['turns'] = turns
        pd_annotation['focus_speaker'] = persona_data[FOCUS_SPEAKER]
        pd_annotation['personality_characteristics'] = {
            "aggreeable": persona_data[AGR],
            "conscientious": persona_data[CON],
            "extroverted": persona_data[EXT],
            "open": persona_data[OPN],
            "neurotic": persona_data[NEU]
        }

        if 'personality_detection' not in d:
            d['personality_detection'] = []
        d['personality_detection'].append(pd_annotation)
    return friends_data

TLiDB_path = "TLiDB_Friends/TLiDB_Friends.json"

# load original Friends data
friends_data = json.load(open(TLiDB_path, "r"))

personality_detection_data = []
with open('emory_personality_detection.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    headers = next(csv_reader)
    for row in csv_reader:
        personality_detection_data.append(row)
        
friends_data = add_personality_detection_annotations(friends_data, personality_detection_data, headers)

with open(TLiDB_path, "w") as f:
    json.dump(friends_data, f, indent=2)