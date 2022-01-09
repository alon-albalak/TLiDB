import json
from tqdm import tqdm
from difflib import SequenceMatcher
from utils import untokenize, remove_notes_from_utt, fuzzy_string_find, OPENERS, CLOSERS

def same_dialogue(d1, d2):
    if len(d1) != len(d2):
        return False
    for i in range(len(d1)):
        d1_utt = ' '.join(remove_notes_from_utt(d1[i]['utterance'].split()))
        d2_utt = ' '.join(remove_notes_from_utt(untokenize(d2[i]['utterance'].split()).split()))
        if d1_utt != d2_utt:
            s = SequenceMatcher(None, d1_utt, d2_utt)
            diff = s.ratio()
            if min(len(d1_utt), len(d2_utt)) > 10 and diff < 0.5:
                return False
    return True

def get_friends_datum_from_qa_data(friends_data, qa_data):
    # fix specific utterances
    if qa_data['title'] == "s02_e14_c01":
        qa_data['paragraphs'][0]['utterances:'][3]['utterance'] = "Nice , nice . Hey I got somethin ' for you . [ hands Chandler an envelope . ]"
    if qa_data['title'] == "s03_e05_c14":
        qa_data['paragraphs'][0]['utterances:'][5]['utterance'] = "( Ross recognises her and goes over to the couch , mouthing ' Oh my God ' )"
    if qa_data['title'] == "s04_e03_c07":
        qa_data['paragraphs'][0]['utterances:'][4]['utterance'] = "( They start kissing again , but are interrupted by the phone . )"
        qa_data['paragraphs'][0]['utterances:'][31]['utterance'] = "( Rachel and Sophie both back out and close the door without saying anything . )"
    if qa_data['title'] == "s04_e17_c03":
        qa_data['paragraphs'][0]['qas'][5]['answers'][0]['answer_text'] = "and then a stewardess comes in"
    if qa_data['title'] == "s04_e23_c02":
        qa_data['paragraphs'][0]['utterances:'][37]['utterance'] = "( Chandler kneels down with his arms spread waiting for his hug . )"

    assert(len(qa_data['paragraphs']) == 1)
    for d in friends_data['data']:
        if d['dialogue_id'] == qa_data['title']:
            # ensure that the dialogue is the same
            if same_dialogue(d['dialogue'], qa_data['paragraphs'][0]['utterances:']):
                return d

def match_string_without_note(s1, s2):
    start = ' '.join(remove_notes_from_utt(s1.split())).find(s2)
    return start

def get_answer_without_note_in_text(text, answer):
    split_text = text.split()
    split_answer = untokenize(answer.split()).split()

    note_starts = [i for i, x in enumerate(split_text) if any(opener in x for opener in OPENERS)]
    note_ends = [i for i, x in enumerate(split_text) if any(closer in x for closer in CLOSERS)]
    if not note_starts:
        return -1, ''

    assert(len(note_starts) == len(note_ends))
    assert(
        all([note_starts[i] < note_ends[i] for i in range(len(note_starts))]) and \
        all([note_starts[i+1] > note_ends[i] for i in range(0, len(note_starts)-1)])
        )

    answer_index_start, answer_index_end = -1, -1
    # iterate through text to find start and end of answer
    text_ind, ans_ind = 0, 0
    for i in range(max(len(split_text), len(split_answer))):
        if split_answer[ans_ind] == split_text[text_ind]:
            if ans_ind == 0:
                answer_index_start = text_ind
            ans_ind += 1
            if ans_ind == len(split_answer):
                answer_index_end = text_ind
                break
        text_ind += 1

    if answer_index_start > -1 and answer_index_end == -1:
        for i in range(answer_index_start, len(split_text)):
            if split_text[i][:-1] == split_answer[-1]:
                answer_index_end = i
                break

    if answer_index_start == -1 or answer_index_end == -1:
        return -1, ''

    a_text = untokenize(split_text[answer_index_start:answer_index_end+1])
    a_start = text.find(a_text)

    return a_start, a_text

def get_answer_string_in_text(text, answer):
    a_text = answer
    a_start = text.find(a_text)

    # if the exact string doesn't match, try other methods
    if a_start == -1:    
        a_text = untokenize(answer.split())
        a_start = text.find(a_text)
    if a_start == -1:
        a_start, a_text = fuzzy_string_find(text, answer)
    if a_start == -1:
        a_start, a_text = fuzzy_string_find(text, untokenize(answer.split()))
    if a_start == -1:
        a_start, a_text = get_answer_without_note_in_text(text, answer)
    if a_start == -1:
        a_start, a_text = get_answer_without_note_in_text(text, untokenize(answer.split()))
    return a_start, a_text

def add_QA_annotations(friends_qa_data, friends_data):
    if "question_answering" not in friends_data['metadata']['tasks']:
        friends_data['metadata']['tasks'].append("question_answering")
        friends_data['metadata']['task_metadata']['question_answering'] = {
            "metrics": ["exact_match", "token_f1"],
            "metric_kwargs": { "token_f1": [
                {"average":"macro"},
                {"average":"micro"},
                ]
            }
        }

    # iteratively add question-answer annotations from friendsQA to original data
    for qa_data in tqdm(friends_qa_data):
        qa_annotation = {}

        # retrieve the original datum
        d = get_friends_datum_from_qa_data(friends_data, qa_data)
        qa_annotation['qas'] = []

        for qa in qa_data['paragraphs'][0]['qas']:
            friends_qa = {"id": qa['id'], "question": untokenize(qa['question'].split()), "answers": []}
            for answer in qa['answers']:

                # find the answer start in utterances
                if answer['is_speaker']:
                    a_start = -1
                    a_text = answer['answer_text']
                elif answer['inner_start'] == 0 and answer['inner_end']+1 <= len(answer['answer_text'].split()):
                    a_start = 0
                    a_text = d['dialogue'][answer['utterance_id']]['utterance']
                else:
                    a_start, a_text = get_answer_string_in_text(d['dialogue'][answer['utterance_id']]['utterance'],
                                answer['answer_text'])
                    assert(a_start != -1), "Could not find answer in text"
                a = {
                    "text": a_text,
                    "answer_start": a_start,
                    "answer_utterance_id": answer['utterance_id'],
                    "is_speaker": answer['is_speaker']
                }
                friends_qa['answers'].append(a)
            qa_annotation['qas'].append(friends_qa)
        if 'question_answering' not in d:
            d['question_answering'] = []
        d['question_answering'].append(qa_annotation)
    return friends_data


TLiDB_path="TLiDB_Friends/TLiDB_Friends.json"

# Load original Friends data
friends_data = json.load(open(TLiDB_path, "r"))

# load FriendsQA annotations
data_partitions = ["train", "dev", "test"]
for p in data_partitions:
    friends_qa_data = json.load(open("emory_question_answering_{}.json".format(p), "r"))
    # add annotations to original data
    friends_data = add_QA_annotations(friends_qa_data['data'], friends_data)

with open(TLiDB_path, 'w') as f:
    json.dump(friends_data, f, indent=4)