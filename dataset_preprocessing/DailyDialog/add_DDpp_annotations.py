import json
from tqdm import tqdm
from utils import untokenize, get_DD_by_ID

import difflib

# Load original DailyDialog data
DD_data = json.load(open('TLiDB_DailyDialog/TLiDB_DailyDialog.json', 'r'))

# load DailyDialogue++ data
DD_pp = []
with open("DDpp_train.json", "r") as f:
    for line in f:
        ddpp = json.loads(line)
        ddpp['id'] = f"train-{ddpp['id']}"
        DD_pp.append(ddpp)
with open("DDpp_dev.json", "r") as f:
    for line in f:
        ddpp = json.loads(line)
        ddpp['id'] = f"dev-{ddpp['id']}"
        DD_pp.append(ddpp)
with open("DDpp_test.json", "r") as f:
    for line in f:
        ddpp = json.loads(line)
        ddpp['id'] = f"test-{ddpp['id']}"
        DD_pp.append(ddpp)

def get_DD_datum_from_DDpp_context(DDpp_context, DD_data):
    # DDpp_context = DDpp_context.replace("'", " ").replace('"',"")
    DDpp_context = DDpp_context.replace("'", " ").replace('"',"").replace(" .",".").replace("  "," ").replace("Mr. : ","").lower()
    for datum in DD_data['data']:
        DD_context = " ".join([turn['utterance'] for turn in datum['dialogue']])
        # DD_context = DD_context.replace("'", " ").replace('"',"").replace(r"\\","").lower()
        DD_context = DD_context.replace("'", " ").replace('"',"").replace(r"\\ ","").replace(" .", ".").replace("  "," ").lower()
        if DDpp_context in DD_context:
            return datum
    return UserWarning("Context not found in DD data")

def get_DD_subset_from_DDpp_context(DDpp_context, DD_datum):
    DD_context = []
    for turn in DD_datum['dialogue']:
        if turn['utterance'][:20] in DDpp_context or turn['utterance'][-20:] in DDpp_context:
            DD_context.append(turn['utterance'])
    return DD_context

def get_DD_turn_subset_from_DDpp_start(DDpp_start,DDpp_len, DD_datum):
    DDpp_start = DDpp_start.replace("'", " ").replace('"',"").replace(" .",".").replace("  "," ").replace("Mr. : ","").lower()
    DD_context_turns = []
    started = False
    found = False
    for turn in DD_datum['dialogue']:
        t = turn['utterance'].replace("'", " ").replace('"',"").replace(r"\\ ","").replace(" .", ".").replace("  "," ").lower()
        if DDpp_start[:20] in t or DDpp_start[-20:] in t:
            started = True
            found = True
        if started and len(DD_context_turns) < DDpp_len:
            DD_context_turns.append(turn['turn_id'])
    if not found:
        raise UserWarning("Context not found in DD data")
    return DD_context_turns

def print_string_diff_DD_DDpp(DD_context, DDpp_context):
    DD_context = DD_context.replace("'", " ").replace('"',"").replace(r"\\ ","").replace(" .", ".").replace("  "," ").lower()
    DDpp_context = DDpp_context.replace("'", " ").replace('"',"").replace(" .",".").replace("  "," ").lower()
    for i, s in enumerate(difflib.ndiff(DD_context, DDpp_context)):
        if s[0] == ' ':
            continue
        elif s[0] == '-':
            print(u'Delete "{}" from position {}'.format(s[-1],i))
        elif s[0] == '+':
            print(u'Add "{}" at position {}'.format(s[-1],i))
    print("SAME DIALOGUE")
    print("DD: ",DD_context)
    print("DD++: ", DDpp_context)
    print()


def add_DDpp_annotations(DD_data, DD_pp, known_DDpp_DD_ID_mapping):

    DD_data['metadata']['tasks'].append('adversarial_response_selection')
    DD_data['metadata']['task_metadata']['adversarial_response_selection'] = {
        'metrics': ['accuracy', 'f1']
    }

    not_found = 0
    total = 0
    for ddpp in tqdm(DD_pp):
        total += 1
        if ddpp['id'] in known_DDpp_DD_ID_mapping[0]:
            DD_datum = get_DD_by_ID(known_DDpp_DD_ID_mapping[1][known_DDpp_DD_ID_mapping[0].index(ddpp['id'])], DD_data)
            
            # uncomment below to see string differences between DD context and DD++ context

            # DDpp_context = untokenize(ddpp['context'])
            # DD_context = get_DD_subset_from_DDpp_context(DDpp_context, DD_datum)
            # DD_context = " ".join(DD_context)
            # print(print_string_diff_DD_DDpp(DD_context, DDpp_context))

        else:
            converted_context = untokenize(ddpp['context'])
            DD_datum = get_DD_datum_from_DDpp_context(converted_context, DD_data)
            if isinstance(DD_datum, UserWarning):
                print(ddpp['id'])
                print("CANT FIND")
                print(converted_context)
                print()
                not_found += 1
                continue
    

        DD_context_turns = get_DD_turn_subset_from_DDpp_start(untokenize([ddpp['context'][0]]), len(ddpp['context']), DD_datum)
        formatted_DDpp = {
            'DDpp_id': ddpp['id'],
            'context_turns': DD_context_turns,
            'positive_responses': ddpp['positive_responses'],
            'random_negative_responses': ddpp['random_negative_responses'],
            'adversarial_negative_responses': ddpp['adversarial_negative_responses'],
            }

        if 'adversarial_response_selection' not in DD_datum['dialogue_metadata']:
            DD_datum['dialogue_metadata']['adversarial_response_selection'] = None
            DD_datum['adversarial_response_selection'] = []
        DD_datum['adversarial_response_selection'].append(formatted_DDpp)

    assert(not_found == 0)

    return DD_data

known_DDpp_DD_ID_mapping = [
    ['train-265','train-561', 'train-585', 'train-641', 'train-976', 'train-1201', 'train-1468',
    'train-1717', 'train-1788', 'train-1848', 'train-2362', 'train-2364', 'train-2470', 'train-2514',
    'train-2515', 'train-2677', 'train-2825','train-2828', 'train-2963', 'train-3340', 'train-3407',
    'train-3477','train-4060', 'train-4278', 'train-4393', 'train-4403', 'train-4763', 'train-4878',
    'train-4969','train-5573','train-5761', 'train-5917', 'train-6044', 'train-6184','train-6381',
    'train-6609', 'train-6737', 'train-6977', 'train-7527', 'train-7959', 'train-8486', 'train-8717',
    'train-8949', 'train-9033', 'dev-7', 'dev-488', 'test-121'],
    ['dialogue-10532','dialogue-4570','dialogue-6445','dialogue-9569','dialogue-363', 'dialogue-2203', 
    'dialogue-10059','dialogue-7378','dialogue-2168','dialogue-4272','dialogue-287','dialogue-3129',
    'dialogue-12629','dialogue-8556','dialogue-245','dialogue-7738','dialogue-3663','dialogue-10631',
    'dialogue-7568','dialogue-6122','dialogue-2186','dialogue-12689','dialogue-6217','dialogue-8430',
    'dialogue-3146','dialogue-1560','dialogue-1462','dialogue-1141','dialogue-10773','dialogue-7944',
    'dialogue-5369','dialogue-1549','dialogue-139','dialogue-11952','dialogue-1172','dialogue-4','dialogue-91',
    'dialogue-426','dialogue-570','dialogue-260','dialogue-5447','dialogue-274','dialogue-199','dialogue-45',
    'dialogue-7638','dialogue-12519','dialogue-12401']
    ]

DD_data = add_DDpp_annotations(DD_data, DD_pp, known_DDpp_DD_ID_mapping)

with open('TLiDB_DailyDialog/TLiDB_DailyDialog.json',"w", encoding='utf8') as f:
    json.dump(DD_data, f, indent=2, ensure_ascii=False)