import json
from tqdm import tqdm
from utils import untokenize, get_DD_by_ID

# Load original DailyDialog data
DD_data = json.load(open('TLiDB_Daily_Dialogue/TLiDB_Daily_Dialogue.json', 'r'))

# load DailyDialogue++ data
DD_pp = []
with open("DDpp_train.json", "r") as f:
    for line in f:
        DD_pp.append(json.loads(line))
with open("DDpp_dev.json", "r") as f:
    for line in f:
        DD_pp.append(json.loads(line))
with open("DDpp_test.json", "r") as f:
    for line in f:
        DD_pp.append(json.loads(line))

def get_DD_datum_from_DDpp_context(DDpp_context, DD_data):
    DDpp_context = DDpp_context.replace("'", " ").replace('"',"")
    for datum in DD_data['data']:
        DD_context = " ".join([turn['utterance'] for turn in datum['dialogue']])
        DD_context = DD_context.replace("'", " ").replace('"',"").replace(r"\\","")
        if DDpp_context in DD_context:
            return datum
    return UserWarning("Context not found in DD data")

def add_DDpp_annotations(DD_data, DD_pp, known_DDpp_DD_ID_mapping):

    not_found = 0
    total = 0
    for ddpp in tqdm(DD_pp):
        total += 1
        if ddpp['id'] in known_DDpp_DD_ID_mapping[0]:
            DD_datum = get_DD_by_ID(known_DDpp_DD_ID_mapping[1][known_DDpp_DD_ID_mapping[0].index(ddpp['id'])], DD_data)
        else:
            converted_context = untokenize(ddpp['context'])
            DD_datum = get_DD_datum_from_DDpp_context(converted_context, DD_data)
            if isinstance(DD_datum, UserWarning):
                not_found += 1
                continue

    print(f"{not_found} contexts not found in DD data")
    print(f"{total} contexts found in DD data")

    return DD_data

known_DDpp_DD_ID_mapping = [
    [265,561, 585, 641, 976, 1201, 1468, 1717, 1788, 1848, 2362, 2364, 2470, 2514, 2515],
    ['dialogue-10532','dialogue-4570','dialogue-6445','dialogue-9569','dialogue-363', 'dialogue-2203', 
    'dialogue-10059','dialogue-7378','dialogue-2168','dialogue-4272','dialogue-287','dialogue-3129',
    'dialogue-12629','dialogue-8556','dialogue-245']
    ]
known_DDpp_DD_ID_mapping= [[],[]]

DD_data = add_DDpp_annotations(DD_data, DD_pp, known_DDpp_DD_ID_mapping)