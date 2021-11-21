import json
import csv
import re
from utils import untokenize, get_DD_by_ID, create_full_DD_dialogue
from tqdm import tqdm

# Load original DailyDialog data
DD_data = json.load(open('TLiDB_Daily_Dialogue/TLiDB_Daily_Dialogue.json', 'r'))

# load CIDER main data
CIDER_data = json.load(open('CIDER_main.json', 'r'))

# load CIDER NLI data
CIDER_DNLI_fields = ["dialogue", "head", "relation", "tail", "label"]
CIDER_DNLI = list(csv.reader(open("CIDER_DNLI_train.tsv", "r"), delimiter='\t'))
CIDER_DNLI.extend(list(csv.reader(open("CIDER_DNLI_test.tsv", "r"), delimiter='\t')))

# load CIDER span extraction data\
CIDER_sp_ex = json.load(open("CIDER_sp_ex_train.json","r"))
CIDER_sp_ex.extend(json.load(open("CIDER_sp_ex_test.json","r")))

def convert_CIDER_dialogue_to_NLI_format(CIDER_dialogue):
    NLI_format = ""
    CIDER_dialogue = CIDER_dialogue.split('    ')
    for utterance in CIDER_dialogue:
        utterance = utterance.replace("A: ", "").replace("B: ", "")
        NLI_format += untokenize([utterance]) + " "
    NLI_format = NLI_format.strip()
    return NLI_format

def convert_CIDER_dialogue_to_DD_format(CIDER_dialogue):
    DD_format_dialogue = []
    CIDER_dialogue = CIDER_dialogue.split('    ')
    for utterance in CIDER_dialogue:
        utterance = utterance.replace("A:", "").replace("B:", "")
        DD_format_dialogue.append(untokenize([utterance]))
    return DD_format_dialogue

def get_CIDER_datum_from_DNLI_datum(DNLI_datum, CIDER_data):
    DNLI_dialogue = untokenize([DNLI_datum[CIDER_DNLI_fields.index('dialogue')]])
    for datum in CIDER_data:
        if datum['DNLI_format'] == DNLI_dialogue:
            return datum
    raise UserWarning(f"Could not find in CIDER data\n\t{DNLI_datum}")

def get_DD_datum_from_DNLI_datum(DNLI_datum, DD_data):
    DNLI_dialogue = untokenize([DNLI_datum[CIDER_DNLI_fields.index('dialogue')]])
    for datum in DD_data['data']:
        DD_dialogue = " ".join([turn['utterance'] for turn in datum['dialogue']])
        if DD_dialogue == DNLI_dialogue:
            return datum
    raise UserWarning(f"Could not find in DD data\n\t{DNLI_datum}")

def get_CIDER_datum_from_sp_ex_datum(sp_ex_datum, CIDER_data):
    sp_ex_dialogue = untokenize([sp_ex_datum['context']])
    for datum in CIDER_data:
        if datum['DNLI_format'] == sp_ex_dialogue:
            return datum
        if datum['DNLI_format'][:50] == sp_ex_dialogue[:50]:
            return datum
    raise UserWarning(f"Could not find in CIDER data\n\t{sp_ex_datum}")

def get_DD_datum_from_sp_ex_datum(sp_ex_datum, DD_data):
    sp_ex_dialogue = untokenize([sp_ex_datum['context']])
    for datum in DD_data['data']:
        DD_dialogue = " ".join([turn['utterance'] for turn in datum['dialogue']])
        if DD_dialogue == sp_ex_dialogue:
            return datum
        if DD_dialogue[:50] == sp_ex_dialogue[:50]:
            return datum
    raise UserWarning(f"Could not find in DD data\n\t{sp_ex_datum}")

def get_DD_datum_from_CIDER_dialogue(CIDER_dialogue, original_DD_data, accept_CIDER_changes=False, reject_CIDER_changes=False,DD_ID=None):
    assert(not (accept_CIDER_changes and reject_CIDER_changes))
    assert(not (accept_CIDER_changes or reject_CIDER_changes) or DD_ID is not None), "If accepting CIDER changes, you must specify the DD datum"
    #it's really inefficient, but because CIDER does not use IDs from DD,
    #   we have to search through the list of original data
    #   to find the original dialogue
    
    formatted_CIDER_dialogue = convert_CIDER_dialogue_to_DD_format(CIDER_dialogue)
    found = False

    if accept_CIDER_changes:
        DD_datum = get_DD_by_ID(DD_ID, original_DD_data)
        for DD_turn, CIDER_turn in zip(DD_datum['dialogue'], formatted_CIDER_dialogue):
            DD_turn['utterance'] = CIDER_turn
        found = True
    elif reject_CIDER_changes:
        DD_datum = get_DD_by_ID(DD_ID, original_DD_data)
        found = True
    else:
        for DD_datum in original_DD_data['data']:
            if DD_datum['dialogue'][0]['utterance'] == formatted_CIDER_dialogue[0]:
                if len(DD_datum['dialogue']) != len(formatted_CIDER_dialogue):
                    continue
                    # raise UserWarning("CIDER dialogue and original dialogue have different lengths")
                unmatched_turns = False
                for DD_turn, CIDER_turn in zip(DD_datum['dialogue'], formatted_CIDER_dialogue):
                    if DD_turn['utterance'] != CIDER_turn:
                        unmatched_turns = True
                        break
                if not unmatched_turns:
                    found = True
                    break
    if not found:
        raise UserWarning(f"Could not find in original data\n\t{formatted_CIDER_dialogue}")
    return DD_datum

def annotate_CIDER_data(original_DD_data, CIDER_data, accepted_CIDER_DD_differences, rejected_CIDER_DD_differences):
    """
    add DD_ID to CIDER_data and convert CIDER dialogue to DNLI format
    """
    
    DD_CIDER_data = []
    # convert all CIDER dialogue to DD format
    for CIDER_datum in tqdm(CIDER_data):
        # skip datums from DREAM or MuTual
        if 'daily-dialogue' not in CIDER_datum['id']:
            continue
        # convert dialogue to DD format
        if CIDER_datum['id'] in accepted_CIDER_DD_differences[0]:
            accept_CIDER_changes = True
            DD_ID = accepted_CIDER_DD_differences[1][accepted_CIDER_DD_differences[0].index(CIDER_datum['id'])]
        elif CIDER_datum['id'] in rejected_CIDER_DD_differences[0]:
            reject_CIDER_changes = True
            DD_ID = rejected_CIDER_DD_differences[1][rejected_CIDER_DD_differences[0].index(CIDER_datum['id'])]
        else:
            accept_CIDER_changes = False
            reject_CIDER_changes = False
            DD_ID = None
        DD_datum = get_DD_datum_from_CIDER_dialogue(CIDER_datum['utterances'], original_DD_data, accept_CIDER_changes, reject_CIDER_changes, DD_ID)
        CIDER_datum['DD_ID'] = DD_datum['dialogue_id']
        CIDER_datum['DNLI_format'] = convert_CIDER_dialogue_to_NLI_format(CIDER_datum['utterances'])
        DD_CIDER_data.append(CIDER_datum)
    return DD_CIDER_data

def add_CIDER_dialogue_nli_annotations(original_DD_data, CIDER_data, DD_CIDER_data, CIDER_DNLI):
    # update data fields
    original_DD_data['metadata']['tasks'].append("dialogue_nli")
    original_DD_data['metadata']['task_metadata']['dialogue_nli'] = {
        'labels':['contradiction','entailment'],'metrics':['f1', 'precision', 'recall']
    }

    found_CIDER = 0
    found_DD = 0
    not_found = 0
    # add NLI annotations to original data
    prev_dialogue = ""
    prev_dialogue_failed = False
    cur_datum = None
    for NLI_datum in tqdm(CIDER_DNLI):
        # if this is the same as the previous dialogue, and that one wasn't in DD, skip it
        if NLI_datum[CIDER_DNLI_fields.index('dialogue')] == prev_dialogue:
            if prev_dialogue_failed:
                continue

        # otherwise, try to find the dialogue in DD
        else:
            prev_dialogue = NLI_datum[CIDER_DNLI_fields.index('dialogue')]

            # first, try to find the dialogue in CIDER
            try:
                cur_datum = get_CIDER_datum_from_DNLI_datum(NLI_datum, DD_CIDER_data)
                found_CIDER += 1
                prev_dialogue_failed = False
            except:
                # next, try original DD data
                try:
                    cur_datum = get_DD_datum_from_DNLI_datum(NLI_datum, original_DD_data)
                    found_DD += 1
                    prev_dialogue_failed = False
                except:
                    not_found += 1
                    prev_dialogue_failed = True

        if 'DD_ID' in cur_datum:
            cur_datum = get_DD_by_ID(cur_datum['DD_ID'], original_DD_data)
        
        if 'dialogue_nli' not in cur_datum:
            cur_datum['dialogue_nli'] = []
        cur_datum['dialogue_nli'].append({
            'head': untokenize([NLI_datum[CIDER_DNLI_fields.index('head')]]),
            'relation': untokenize([NLI_datum[CIDER_DNLI_fields.index('relation')]]),
            'tail': untokenize([NLI_datum[CIDER_DNLI_fields.index('tail')]]),
            'label': untokenize([NLI_datum[CIDER_DNLI_fields.index('label')]])})

    assert found_CIDER == 228
    assert found_DD== 17
    assert not_found == 561

    return original_DD_data

def add_CIDER_span_extraction_annotations(DD_data, CIDER_data, DD_CIDER_data, CIDER_sp_ex):
    # update data fields
    DD_data['metadata']['tasks'].append("dialogue_reasoning_span_extraction")
    DD_data['metadata']['task_metadata']['dialogue_reasoning_span_extraction'] = {'metrics':['exact_match','token_f1','no_match']}
    
    found_CIDER = 0
    found_DD = 0
    not_found = 0
    cur_datum = None
    cur_ID = None
    DD_sp_ex = None
    num_missing_answer = 0
    
    # iteratively add span extraction annotations to original data
    for sp_ex in tqdm(CIDER_sp_ex):
        # skip datums from DREAM or MuTual
        if 'daily-dialogue' not in sp_ex['qas'][0]['id']:
            continue


        # first, check if the datum has previously been annotated with the DD_ID
        DD_ID_found = False
        sp_ex_ID = "-".join(sp_ex['qas'][0]['id'].split('-')[:3])
        if sp_ex_ID != cur_ID:
            cur_ID = sp_ex_ID
            if DD_sp_ex:
                cur_datum['dialogue_reasoning_span_extraction'] = DD_sp_ex

            for d in DD_CIDER_data:
                if sp_ex_ID == d['id']:
                    DD_ID = d['DD_ID']
                    DD_ID_found = True
                    break
            if DD_ID_found and 'DD_ID' in d:
                cur_datum = get_DD_by_ID(DD_ID, DD_data)
                found_CIDER += 1
            else:
                
                # if not, try to find the dialogue in CIDER
                try:
                    cur_datum = get_CIDER_datum_from_sp_ex_datum(sp_ex, DD_CIDER_data)
                    found_CIDER += 1
                except:
                    # next, try original DD data
                    try:
                        cur_datum = get_DD_datum_from_sp_ex_datum(sp_ex, DD_data)
                        found_DD += 1
                    except:
                        not_found += 1
                        continue
    
            full_dialogue = create_full_DD_dialogue(cur_datum)
            DD_sp_ex = {'context':full_dialogue,'qas':[]}

        # compile the span extraction annotations in our format
        for qa in sp_ex['qas']:
            DD_qa = {'question':qa['question'], 'answers':[]}
            for answer in qa['answers']:
                text = untokenize([answer['text']])
                answer_start = DD_sp_ex['context'].find(text)
                # for some reason in CIDER, they sometimes create synthetic answers that are not in the context
                if answer_start == -1:
                    num_missing_answer += 1
                    DD_sp_ex['context'] += f" {text}"
                    answer_start = DD_sp_ex['context'].find(text)
                DD_qa['answers'].append({'text':text, 'answer_start':answer_start})
            DD_sp_ex['qas'].append(DD_qa)
            
    assert found_CIDER == 243
    assert found_DD == 0
    assert not_found == 0

    return DD_data


# Known differences between CIDER and DD to accept (typos, etc.)
accepted_CIDER_DD_differences = [
        ['daily-dialogue-0020', 'daily-dialogue-0409', 'daily-dialogue-1010', 'daily-dialogue-1171'],
        ['dialogue-260', 'dialogue-813', 'dialogue-5369', 'dialogue-570']]
# Known differences between CIDER and DD to reject
rejected_CIDER_DD_differences = [
        ['daily-dialogue-0072', 'daily-dialogue-0125'],
        ['dialogue-585','dialogue-6167']]

DD_CIDER_data = annotate_CIDER_data(DD_data, CIDER_data, accepted_CIDER_DD_differences, rejected_CIDER_DD_differences)
DD_data = add_CIDER_dialogue_nli_annotations(DD_data, CIDER_data, DD_CIDER_data, CIDER_DNLI)
DD_date = add_CIDER_span_extraction_annotations(DD_data, CIDER_data, DD_CIDER_data, CIDER_sp_ex)

with open('TLiDB_Daily_Dialogue/TLiDB_Daily_Dialogue.json',"w", encoding='utf8') as f:
    json.dump(DD_data, f, indent=2, ensure_ascii=False)