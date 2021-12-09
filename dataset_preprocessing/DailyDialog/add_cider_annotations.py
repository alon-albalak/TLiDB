import json
import csv
import re
from utils import untokenize, get_DD_by_ID, create_full_DD_dialogue
from tqdm import tqdm

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
    raise UserWarning(f"Could not find in CIDER data\n\t{sp_ex_datum}")

def get_DD_datum_from_sp_ex_datum(sp_ex_datum, DD_data):
    sp_ex_dialogue = untokenize([sp_ex_datum['context']])
    for datum in DD_data['data']:
        DD_dialogue = " ".join([turn['utterance'] for turn in datum['dialogue']])
        if DD_dialogue == sp_ex_dialogue:
            return datum
    raise UserWarning(f"Could not find in DD data\n\t{sp_ex_datum}")

def get_CIDER_datum_from_mcq_datum(mcq_datum, CIDER_data):
    raise NotImplementedError

def get_DD_datum_from_mcq_datum(mcq_datum, DD_data):
    raise NotImplementedError

def get_CIDER_datum_from_crp_datum(crp_datum, CIDER_data):
    crp_dialogue = untokenize([crp_datum[CIDER_CRP_fields.index('context')]])
    for datum in CIDER_data:
        if datum['DNLI_format'] == crp_dialogue:
            return datum
    raise UserWarning(f"Could not find in CIDER data\n\t{crp_datum}")

def get_DD_datum_from_crp_datum(crp_datum, DD_data):
    crp_dialogue = untokenize([crp_datum[CIDER_CRP_fields.index('context')]])
    for datum in DD_data['data']:
        DD_dialogue = " ".join([turn['utterance'] for turn in datum['dialogue']])
        if DD_dialogue == crp_dialogue:
            return datum
    raise UserWarning(f"Could not find in DD data\n\t{crp_datum}")

def get_DD_datum_from_CIDER_dialogue(CIDER_dialogue, original_DD_data, accept_CIDER_changes=False, reject_CIDER_changes=False,DD_ID=None):
    assert(not (accept_CIDER_changes and reject_CIDER_changes))
    assert(not (accept_CIDER_changes or reject_CIDER_changes) or DD_ID is not None), "If accepting CIDER changes, you must specify the DD datum"

    
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
                # some dialogues begin with the same first utterance, but diverge after that
                if len(DD_datum['dialogue']) != len(formatted_CIDER_dialogue):
                    continue
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

def add_CIDER_dialogue_nli_annotations(original_DD_data, DD_CIDER_data, CIDER_DNLI, partition):
    # update data fields
    if "dialogue_nli" not in original_DD_data['metadata']['tasks']:
        original_DD_data['metadata']['tasks'].append("dialogue_nli")
        original_DD_data['metadata']['task_metadata']['dialogue_nli'] = {
            'labels':['contradiction','entailment'],'metrics':['f1', 'precision', 'recall'],
            "metric_kwargs":{"f1":[{"average":"micro"},{"average":"weighted"}]}
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
            cur_datum['dialogue_metadata']['dialogue_nli'] = {"original_data_partition":partition}
        assert(cur_datum['dialogue_metadata']['dialogue_nli']['original_data_partition'] == partition)
        cur_datum['dialogue_nli'].append({
            'head': untokenize([NLI_datum[CIDER_DNLI_fields.index('head')]]),
            'relation': untokenize([NLI_datum[CIDER_DNLI_fields.index('relation')]]),
            'tail': untokenize([NLI_datum[CIDER_DNLI_fields.index('tail')]]),
            'label': untokenize([NLI_datum[CIDER_DNLI_fields.index('label')]])})

    return original_DD_data, found_CIDER, found_DD, not_found

def add_CIDER_span_extraction_annotations(DD_data, DD_CIDER_data, CIDER_sp_ex):
    # update data fields
    if "dialogue_readability" not in DD_data['metadata']['tasks']:
        DD_data['metadata']['tasks'].append("dialogue_reasoning_span_extraction")
        DD_data['metadata']['task_metadata']['dialogue_reasoning_span_extraction'] = {
            'metrics':['exact_match','token_f1','no_match'],
            "metric_kwargs":{
                "exact_match":{"ignore_phrases":["impossible"]},
                "token_f1":{"ignore_phrases":["impossible"]},
            }
        }
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
            if 'dialogue_reasoning_span_extraction' not in cur_datum['dialogue_metadata']:
                cur_datum['dialogue_metadata']['dialogue_reasoning_span_extraction'] = None
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

def mcq_in_DD_MCQ(q, options, label, DD_MCQ):
    for mcq in DD_MCQ['mcqs']:
        if mcq['question'] == q and mcq['options'] == options and mcq['label'] == label:
            return True
    return False

def add_CIDER_multiple_choice_span_selection_annotations(DD_data, DD_CIDER_data, CIDER_MCQ):
    DD_data['metadata']['tasks'].append("dialogue_reasoning_multiple_choice_span_selection")
    DD_data['metadata']['task_metadata']['dialogue_reasoning_multiple_choice_span_selection'] = {'metrics':['accuracy']}
    
    found_CIDER = 0
    found_DD = 0
    not_found = 0
    cur_datum = None
    cur_ID = None
    DD_MCQ = None
    duplicate_mcqs = 0

    # iteratively add span selection annotations to original data
    for mcq in tqdm(CIDER_MCQ):
        # skip datums from DREAM or MuTual
        if 'daily-dialogue' not in mcq[CIDER_MCQ_fields.index('CIDER_ID')]:
            continue
        # first, check if the datum has previously been annotated with the DD_ID
        DD_ID_found = False
        mcq_ID = "-".join(mcq[CIDER_MCQ_fields.index('CIDER_ID')].split('-')[:3])
        if mcq_ID != cur_ID:
            cur_ID = mcq_ID
            if DD_MCQ:
                cur_datum['dialogue_reasoning_multiple_choice_span_selection'] = DD_MCQ

            for d in DD_CIDER_data:
                if mcq_ID == d['id']:
                    DD_ID = d['DD_ID']
                    DD_ID_found = True
                    break
            if DD_ID_found and 'DD_ID' in d:
                cur_datum = get_DD_by_ID(DD_ID, DD_data)
                found_CIDER += 1
            else:
                # if not, try to find the dialogue in CIDER
                try:
                    cur_datum = get_CIDER_datum_from_mcq_datum(mcq, DD_CIDER_data)
                    found_CIDER += 1
                except Exception as e:
                    # next, try original DD data
                    try:
                        cur_datum = get_DD_datum_from_mcq_datum(mcq, DD_data)
                        found_DD += 1
                    except Exception as e:
                        not_found += 1
                        raise e

            full_dialogue = create_full_DD_dialogue(cur_datum)
            cur_datum['dialogue_metadata']['dialogue_reasoning_multiple_choice_span_selection'] = None
            DD_MCQ = {'context':full_dialogue,'mcqs':[]}

        # compile the span selection annotations in our format
        q = mcq[CIDER_MCQ_fields.index('question')]
        options = [untokenize([mcq[CIDER_MCQ_fields.index(f'option{i}')]]) for i in range(4)]
        label = mcq[CIDER_MCQ_fields.index('label')]
        is_duplicate_mcq = mcq_in_DD_MCQ(q, options, label, DD_MCQ)
        if not is_duplicate_mcq:
            DD_MCQ['mcqs'].append({'question':q, 'options':options, 'label':label})
        else:
            duplicate_mcqs += 1

    assert(found_CIDER == 241)
    assert(found_DD == 0)
    assert(not_found == 0)
    assert(duplicate_mcqs == 56)
    
    return DD_data

def add_CIDER_commonsense_relation_prediction_annotations(original_DD_data, DD_CIDER_data, CIDER_CRP, partition):
    original_DD_data['metadata']['tasks'].append("dialogue_reasoning_commonsense_relation_prediction")
    original_DD_data['metadata']['task_metadata']['dialogue_reasoning_commonsense_relation_prediction'] = {
        'labels':CIDER_CRP_labels, 'metrics':['accuracy']
        }
    
    found_CIDER = 0
    found_DD = 0
    not_found = 0
    cur_datum = None

    prev_dialogue = ""
    prev_dialogue_failed = False

    # iteratively add span selection annotations to original data
    for crp in tqdm(CIDER_CRP):
        
        # if this is the same as the previous dialogue, and that one wasn't in DD, skip it
        if crp[CIDER_CRP_fields.index("context")] == prev_dialogue:
            if prev_dialogue_failed:
                continue

        # otherwise try to find the dialogue in DD
        else:
            prev_dialogue = crp[CIDER_CRP_fields.index("context")]

            # first, try to find the dialogue in CIDER
            try:
                cur_datum = get_CIDER_datum_from_crp_datum(crp, DD_CIDER_data)
                found_CIDER += 1
                prev_dialogue_failed = False
            except:
                # next, try original DD data
                try:
                    cur_datum = get_DD_datum_from_crp_datum(crp, original_DD_data)
                    found_DD += 1
                    prev_dialogue_failed = False
                except:
                    not_found += 1
                    prev_dialogue_failed = True


            if 'DD_ID' in cur_datum:
                cur_datum = get_DD_by_ID(cur_datum['DD_ID'], original_DD_data)
        
        if 'dialogue_reasoning_commonsense_relation_prediction' not in cur_datum:
            cur_datum['dialogue_reasoning_commonsense_relation_prediction'] = []
            cur_datum['dialogue_metadata']['dialogue_reasoning_commonsense_relation_prediction'] = {"original_data_partition":partition}

        head,tail = crp[CIDER_CRP_fields.index("entities")].split("[SEP]")
        cur_datum['dialogue_reasoning_commonsense_relation_prediction'].append({
            'head': head.strip(),
            'tail': tail.strip(),
            'relation': crp[CIDER_CRP_fields.index("relation")]
        })
        assert(cur_datum['dialogue_metadata']['dialogue_reasoning_commonsense_relation_prediction']['original_data_partition'] == partition)
        


    return original_DD_data, found_CIDER, found_DD, not_found

# Load original DailyDialog data
DD_data = json.load(open('TLiDB_DailyDialog/TLiDB_DailyDialog.json', 'r'))

# load CIDER main data
CIDER_data = json.load(open('CIDER_main.json', 'r'))

# Known differences between CIDER and DD to accept (typos, etc.)
accepted_CIDER_DD_differences = [
        ['daily-dialogue-0020', 'daily-dialogue-0409', 'daily-dialogue-1010', 'daily-dialogue-1171'],
        ['dialogue-260', 'dialogue-813', 'dialogue-5369', 'dialogue-570']]
# Known differences between CIDER and DD to reject
rejected_CIDER_DD_differences = [
        ['daily-dialogue-0072', 'daily-dialogue-0125'],
        ['dialogue-585','dialogue-6167']]

DD_CIDER_data = annotate_CIDER_data(DD_data, CIDER_data, accepted_CIDER_DD_differences, rejected_CIDER_DD_differences)


data_partitions = ["train","test"]

# load CIDER NLI data
CIDER_DNLI_fields = ["dialogue", "head", "relation", "tail", "label"]
found_CIDER, found_DD, not_found = 0, 0, 0
for p in data_partitions:
    CIDER_DNLI = list(csv.reader(open(f"CIDER_DNLI_{p}.tsv", "r"), delimiter="\t"))
    DD_data, f_CIDER, f_DD, nf = add_CIDER_dialogue_nli_annotations(DD_data, DD_CIDER_data, CIDER_DNLI, p)
    found_CIDER += f_CIDER
    found_DD += f_DD
    not_found += nf
# make sure we found all the CIDER data
assert found_CIDER == 228
assert found_DD== 17
assert not_found == 561

# load CIDER span extraction data
# CIDER span extraction datums are not split by dialogue, some datums from training and testing come from the same underlying DD dialogue
CIDER_sp_ex = json.load(open("CIDER_sp_ex_train.json","r"))
CIDER_sp_ex.extend(json.load(open("CIDER_sp_ex_test.json","r")))
DD_data = add_CIDER_span_extraction_annotations(DD_data, DD_CIDER_data, CIDER_sp_ex)

# load CIDER multiple choice span selection
# CIDER MCQ datums are not split by dialogue, some datums from training and validation come from the same underlying DD dialogue
CIDER_MCQ_fields = ["_", "CIDER_ID", "fold", "question", "context", "_", "_", "option0","option1","option2","option3","label"]
CIDER_MCQ = list(csv.reader(open("CIDER_MCQ_train_iter0.csv", "r")))[1:]
CIDER_MCQ.extend(list(csv.reader(open("CIDER_MCQ_train_iter35.csv", "r")))[1:])
CIDER_MCQ.extend(list(csv.reader(open("CIDER_MCQ_val_iter0.csv", "r")))[1:])
CIDER_MCQ.extend(list(csv.reader(open("CIDER_MCQ_val_iter35.csv", "r")))[1:])
CIDER_MCQ = sorted(CIDER_MCQ, key=lambda x: x[CIDER_MCQ_fields.index('CIDER_ID')])
DD_data = add_CIDER_multiple_choice_span_selection_annotations(DD_data, DD_CIDER_data, CIDER_MCQ)

# load CIDER commonsense relation prediction data
CIDER_CRP_labels = sorted(open("CIDER_RP_relations.txt", "r").read().splitlines())
CIDER_CRP_fields = ["_", "context", "entities", "relation"]
for p in data_partitions:
    CIDER_CRP = list(csv.reader(open(f"CIDER_RP_{p}.csv", "r")))[1:]
    DD_data, f_CIDER, f_DD, nf = add_CIDER_commonsense_relation_prediction_annotations(DD_data, DD_CIDER_data, CIDER_CRP, p)
# make sure we found all the CIDER data
assert found_CIDER == 228
assert found_DD== 17
assert not_found == 561

with open('TLiDB_DailyDialog/TLiDB_DailyDialog.json',"w", encoding='utf8') as f:
    json.dump(DD_data, f, indent=2, ensure_ascii=False)