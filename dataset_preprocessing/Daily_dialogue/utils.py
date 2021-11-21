import re
def untokenize(words):
    """
    Untokenizing a text undoes the tokenizing operation, restoring
    punctuation and spaces to the places that people expect them to be.
    Ideally, `untokenize(tokenize(text))` should be identical to `text`,
    except for line breaks.
    """
    text = ' '.join(words)
    step1 = text.replace("。",".").replace("’","'").replace("`` ", '"').replace(" ''", '"').replace('. . .', '...').replace(" ` ", " '")
    step2 = step1.replace("( ","(").replace(" ( ", " (").replace(" )", ")").replace(" ) ", ") ").replace(" -- ", " - ").replace("—","-").replace("–","-").replace('”','"').replace('“','"').replace("‘","'").replace("’","'")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = re.sub(r'(?<=[.,])(?=[^\s])', r' ', step4)
    step6 = step5.replace(" '", "'").replace(" n't", "n't").replace("n' t", "n't").replace("t' s","t's").replace("' ll", "'ll").replace("I' m", "I'm").replace(
        "can not", "cannot").replace("I' d", "I'd").replace("' re", "'re")
    return step6.strip()

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

def get_DD_by_ID(DD_ID, DD_data):
    for d in DD_data['data']:
        if d['dialogue_id'] == DD_ID:
            return d
    raise UserWarning("Could not find dialogue with ID: " + DD_ID)

def create_full_DD_dialogue(datum):
    dialogue = ""
    for turn in datum['dialogue']:
        dialogue += turn['speakers'][0]
        dialogue += ': '
        dialogue += turn['utterance']
        dialogue += ' '
    return dialogue[:-1]
