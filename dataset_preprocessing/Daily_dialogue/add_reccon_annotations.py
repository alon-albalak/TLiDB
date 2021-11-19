import os
import json
from utils import untokenize

# Load original DailyDialog data
DD_data = json.load(open('TLiDB_Daily_Dialogue/TLiDB_Daily_Dialogue.json', 'r'))

# update original data fields
DD_data['metadata']['tasks'].append("causal_emotion_span_extraction")
DD_data['metadata']['task_metadata']['causal_emotion_span_extraction'] = {"metrics":["exact_match", "token_f1"]}

# Load RECCON specific data
RECCON_train = json.load(open('RECCON_train.json', 'r'))
RECCON_validation = json.load(open('RECCON_validation.json', 'r'))
RECCON_test = json.load(open('RECCON_test.json', 'r'))
RECCON_data = RECCON_train
RECCON_data.update(RECCON_validation)
RECCON_data.update(RECCON_test)

def convert_REC_ID_to_DD_ID(REC_ID):
    """
    Convert IDs as they are in RECCON into original Daily Dialog IDs
    """
    split, id = REC_ID.split('_')
    id = str(int(id) + 1)
    if split == 'tr':
        return 'dialogue-'+id
    elif split == 'va':
        return 'dialogue-'+str(int(id)+11118)
    assert(split=='te')
    return 'dialogue-'+str(int(id)+12118)

def untokenize_REC_dialogue(dialogue):
    for turn in dialogue:
        turn['utterance'] = untokenize(turn['utterance'].split())
num_emotion_updates = 0
for REC_ID, REC_dialogue in RECCON_data.items():
    DD_ID = convert_REC_ID_to_DD_ID(REC_ID)

    #it's really inefficient, but we have to search through the \
    #   list of original data to find the right dialogue
    found = False
    for d in DD_data['data']:
        if d['dialogue_id'] == DD_ID:
            found = True
            break
    assert(found), f"Could not find {DD_ID} in original data"

    untokenize_REC_dialogue(REC_dialogue[0])

    # fix typos, errors, etc.
    if DD_ID == "dialogue-3186":
        d['dialogue'][11]['utterance'] = "Yes, and I took many pictures."
    if DD_ID == "dialogue-3072":
        d['dialogue'][3]['utterance'] = "Don't judge a book by its cover. Do you know what it's about?"
    if DD_ID == "dialogue-12401":
        d['dialogue'][2]['utterance'] = "Then you will be happy to hear that today all our pizzas are on sale. Two for one."
    if DD_ID == "dialogue-12923":
        d['dialogue'][10]['utterance'] = REC_dialogue[0][10]['utterance']
        for i in range(11,16):
            d['dialogue'][i]['utterance'] = REC_dialogue[0][i]['utterance']
            d['dialogue'][i]['emotion_recognition'] = d['dialogue'][i+1]['emotion_recognition']
            d['dialogue'][i]['dialogue_actions'] = d['dialogue'][i+1]['dialogue_actions']
        d['dialogue'] = d['dialogue'][:-1]
    if DD_ID == "dialogue-3814":
        d['dialogue'][4]['utterance'] = "Here we go. There is a museum of the Beijing Opera art."


    # make sure that the dialogue is identical between DD and RECCON
    # accept updated emotions from RECCON
    for i in range(len(REC_dialogue[0])):
        if REC_dialogue[0][i]['utterance'] != d['dialogue'][i]['utterance']:
            print(REC_dialogue[0][i]['utterance'])
            print(d['dialogue'][i]['utterance'])
            print(len(REC_dialogue[0][i]['utterance']))
            print(len(d['dialogue'][i]['utterance']))
            print(DD_ID)
            print(REC_ID)
            assert(REC_dialogue[0][i]['utterance'] == d['dialogue'][i]['utterance'])
        if REC_dialogue[0][i]['emotion'] != d['dialogue'][i]['emotion_recognition']:
            print(f"Updated \"{REC_dialogue[0][i]['utterance']}\" from {d['dialogue'][i]['emotion_recognition']} to {REC_dialogue[0][i]['emotion']}")
            d['dialogue'][i]['emotion_recognition'] = REC_dialogue[0][i]['emotion']
            num_emotion_updates += 1

    # update DD with RECCON annotations
    d['dialogue_metadata']['causal_emotion_span_extraction'] = {"RECCON_dialogue_id":REC_ID}
    ev_str = 'expanded emotion cause evidence'
    sp_str = 'expanded emotion cause span'
    for REC_turn, DD_turn in zip(REC_dialogue[0], d['dialogue']):
        # TODO: add ev_str turn and sp_str span to DD

        # Neither of the below situations occur
        if ev_str in REC_turn and sp_str not in REC_turn:
            a=1
        if sp_str in REC_turn and ev_str not in REC_turn:
            a=1

print(f"Updated emotions on {num_emotion_updates} utterances")