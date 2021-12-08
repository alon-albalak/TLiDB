import json
import csv
import re
from utils import untokenize, convert_REC_ID_to_DD_ID
from tqdm import tqdm

# Load original DailyDialog data
DD_data = json.load(open('TLiDB_DailyDialog/TLiDB_DailyDialog.json', 'r'))

def get_DD_datum_from_RECCON_ID(REC_ID, original_DD_data):
    id = re.search(f"dailydialog_(.*?)_utt",REC_ID).group(1)
    DD_ID = convert_REC_ID_to_DD_ID(id)

    #it's really inefficient, but we have to search through the \
    #   list of original data to find the right dialogue
    found = False
    for d in original_DD_data['data']:
        if d['dialogue_id'] == DD_ID:
            found = True
            break
    if not found:
        raise UserWarning(f"Could not find {DD_ID} in original data")
    return d

def convert_REC_context_to_DD(REC_context, DD_context):
    # convert RECCON context to DD structure
    # basically, just add speakers
    new_context = ""
    for turn in DD_context:
        if turn['utterance'] in REC_context:
            new_context+= f"{turn['speakers'][0]}: {turn['utterance']} "
    return new_context[:-1]

def add_RECCON_span_extraction_annotations(RECCON_span_extraction_data, original_DD_data, partition):
    # update original data fields
    if "causal_emotion_span_extraction" not in original_DD_data['metadata']['tasks']:
        original_DD_data['metadata']['tasks'].append("causal_emotion_span_extraction")
        original_DD_data['metadata']['task_metadata']['causal_emotion_span_extraction'] = {
            "metrics":["exact_match", "token_f1"],
            "metric_kwargs":{
                "exact_match":{"ignore_phrases":["impossible"]},
                "token_f1":{"ignore_phrases":["impossible"]}}
            }
    # iteratively add span extraction data from RECCON to original data
    for sp_ex in tqdm(RECCON_span_extraction_data):
        DD_sp_ex = {}
        # find original datum by ID
        d = get_DD_datum_from_RECCON_ID(sp_ex['qas'][0]['id'], original_DD_data)
        d['dialogue_metadata']['causal_emotion_span_extraction'] = {"original_data_partition":partition}

        # add speakers to context and tokenize in our method
        DD_sp_ex['context'] = convert_REC_context_to_DD(untokenize([sp_ex['context']]), d['dialogue'])
        DD_sp_ex['qas'] = []
        for qa in sp_ex['qas']:
            DD_qa = {'id':qa['id'], 'is_impossible':qa['is_impossible']}
            
            # pull apart the original question so that we can tokenize it in our method and add speakers
            target_utterance = re.search(r"The target utterance is (.*?) The evidence utterance is", qa['question']).group(1)
            evidence_utterance = re.search(r"The evidence utterance is (.*?) What is the causal span", qa['question']).group(1)
            q_phrase = qa['question'][qa['question'].rfind(evidence_utterance)+len(evidence_utterance)+1:]

            # tokenize
            target_utterance = untokenize([target_utterance])
            evidence_utterance = untokenize([evidence_utterance])
            q_phrase = untokenize([q_phrase])

            # add speakers
            target_utterance = convert_REC_context_to_DD(target_utterance, d['dialogue'])
            evidence_utterance = convert_REC_context_to_DD(evidence_utterance, d['dialogue'])

            # compile the question
            question = f"The target utterance is {target_utterance} The evidence utterance is {evidence_utterance} {q_phrase}"
            DD_qa['question'] = question
            DD_qa['answers'] = []
            # add answers in our format
            for answer in qa['answers']:
                a_text = untokenize([answer['text']])
                if a_text not in DD_sp_ex['context'] and answer['text'] != "Impossible ≠æ≠æ≠ answer":
                    raise UserWarning("Could not find the answer in the context")
                if answer['text'] == "Impossible ≠æ≠æ≠ answer":
                    a_text = "Impossible answer"
                a_start = DD_sp_ex['context'].find(a_text)
                a = {'text':a_text, "answer_start":a_start}
                DD_qa['answers'].append(a)
            DD_sp_ex['qas'].append(DD_qa)
        if 'causal_emotion_span_extraction' not in d:
            d['causal_emotion_span_extraction'] = []
        d['causal_emotion_span_extraction'].append(DD_sp_ex)
    return original_DD_data

def add_RECCON_entailment_annotations(RECCON_entailment_data, RECCON_entailment_fields, original_DD_data, partition):
    if "causal_emotion_entailment" not in original_DD_data['metadata']['tasks']:
        original_DD_data['metadata']['tasks'].append('causal_emotion_entailment')
        original_DD_data['metadata']['task_metadata']['causal_emotion_entailment'] = {
            'labels': ["not entailed","entailed"], 'metrics':['f1', 'accuracy'],
            "metric_kwargs":{"f1":[{"average":"micro"},{"average":"macro"}]}}

    label_mapping = {'0':'not entailed', '1':'entailed'}

    for ent_datum in tqdm(RECCON_entailment_data):
        d = get_DD_datum_from_RECCON_ID(ent_datum[RECCON_entailment_fields.index('id')], original_DD_data)
        d['dialogue_metadata']['causal_emotion_entailment'] = {"original_data_partition":partition}
        DD_ent = {'labels':label_mapping[ent_datum[RECCON_entailment_fields.index("labels")]]}

        emo,target,causal_utt,hist = ent_datum[RECCON_entailment_fields.index("text")].split(" <SEP> ")
        target = convert_REC_context_to_DD(untokenize([target]),d['dialogue'])
        causal_utt = convert_REC_context_to_DD(untokenize([causal_utt]), d['dialogue'])
        hist = convert_REC_context_to_DD(untokenize([hist]), d['dialogue'])

        DD_ent['emotion'] = emo
        DD_ent['target_utterance'] = target
        DD_ent['causal_utterance'] = causal_utt
        DD_ent['history'] = hist

        DD_ent['text'] = " <SEP> ".join([emo,target,causal_utt,hist])

        if 'causal_emotion_entailment' not in d:
            d['causal_emotion_entailment'] = []
        d['causal_emotion_entailment'].append(DD_ent)
    return original_DD_data


data_partitions = [["train","train"],["validation","dev"],["test","test"]]
# load RECCON span extraction
for p in data_partitions:
    RECCON_sp_ex = json.load(open(f"RECCON_span_extraction_{p[0]}.json","r"))
    DD_data = add_RECCON_span_extraction_annotations(RECCON_sp_ex, DD_data,p[1])
# RECCON_sp_ex = json.load(open("RECCON_span_extraction_train.json", 'r'))
# RECCON_sp_ex.extend(json.load(open("RECCON_span_extraction_validation.json", 'r')))
# RECCON_sp_ex.extend(json.load(open("RECCON_span_extraction_test.json", 'r')))

# load RECCON entailment
for p in data_partitions:
    RECCON_ent = list(csv.reader(open(f"RECCON_entailment_{p[0]}.csv", 'r')))
    RECCON_ent_fields = RECCON_ent[0]
    RECCON_ent = RECCON_ent[1:]
    DD_data = add_RECCON_entailment_annotations(RECCON_ent, RECCON_ent_fields, DD_data, p[1])

# RECCON_ent = list(csv.reader(open("RECCON_entailment_train.csv", "r")))
# RECCON_ent.extend(list(csv.reader(open("RECCON_entailment_validation.csv", "r")))[1:])
# RECCON_ent.extend(list(csv.reader(open("RECCON_entailment_test.csv", "r")))[1:])

# DD_data = add_RECCON_span_extraction_annotations(RECCON_sp_ex, DD_data)
# DD_data = add_RECCON_entailment_annotations(RECCON_ent, RECCON_ent_fields, DD_data)

with open('TLiDB_DailyDialog/TLiDB_DailyDialog.json',"w", encoding='utf8') as f:
    json.dump(DD_data, f, indent=2, ensure_ascii=False)