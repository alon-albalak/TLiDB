import re

OPENERS = ["(","[","{"]
CLOSERS = [")","]","}"]

def untokenize(words):
    """
    Untokenizing a text undoes the tokenizing operation, restoring
    punctuation and spaces to the places that people expect them to be.
    Ideally, `untokenize(tokenize(text))` should be identical to `text`,
    except for line breaks.
    """

    # standardize long pauses to "..."
    for i in range(len(words)):
        match = re.sub(r'\.{2,}', r'...', words[i])
        if match != words[i]:
            words[i] = match

    text = ' '.join(words)
    step1 = text.replace("。",".").replace("’","'").replace("`` ", '"').replace(" ''", '"').replace(" ` ", " '").replace(" ,",",")
    step2 = step1.replace(" -- ", " - ").replace("—","-").replace("–","-").replace('”','"').replace('“','"').replace("‘","'").replace("’","'")
    step3 = step2.replace(" '", "'").replace(" n't", "n't").replace("n' t", "n't").replace("t' s","t's").replace("' ll", "'ll").replace("I' m", "I'm").replace(
        "can not", "cannot").replace("I' d", "I'd").replace("' re", "'re").replace("t ' s", "t's").replace("e' s", "e's")
    step4 = step3.replace("? !", "?!").replace("! !", "!!").replace("! ?", "!?").replace("n'y","n't").replace('yarning','yawning').replace(" om V", " on V")
    step5 = step4.replace('. . .', '...').replace("wan na", "wanna").replace("gon na", "gonna")
    step6 = re.sub(r'(?<=[a-zA-Z])\s+(?=[.,:;\?!])', r'', step5)
    step7 = re.sub(r'(\S)(\.{3})', r'\1 \2', step6)
    step8 = re.sub(r'(\.{3})(\S)', r'\1 \2', step7)
    return step8.strip()

def remove_notes_from_utt(utterance):
    new_utt = []
    note_starts = [i for i, x in enumerate(utterance) if any(opener in x for opener in OPENERS)]
    note_ends = [i for i, x in enumerate(utterance) if any(closer in x for closer in CLOSERS)]
    if note_starts:
        assert(len(note_starts) == len(note_ends))
        assert(all([note_starts[i] <= note_ends[i] for i in range(len(note_starts))]))
        excluded_indices = []
        for s,e in zip(note_starts, note_ends):
            excluded_indices.extend(range(s,e+1))
        for i,x in enumerate(utterance):
            if i not in excluded_indices:
                new_utt.append(x)
    else:
        new_utt = utterance
    return new_utt

def fuzzy_string_find(string, pattern, pattern_divisor=10, min_portion=2):
    """
    Finds the pattern in string by only looking at the first and last portions of pattern
    """

    pattern_length = len(pattern.split())

    portion = max(pattern_length//pattern_divisor, min_portion)

    first_portion = pattern.split()[:portion]
    last_portion = pattern.split()[-portion:]
    first_portion = ' '.join(first_portion)
    last_portion = ' '.join(last_portion)
    first_start = string.find(first_portion)
    last_start = string.rfind(last_portion)
    
    if first_start > -1 and last_start > -1:
        last_end = last_start + len(last_portion)
        return first_start, string[first_start:last_end]
    else:
        return -1, ''

def remove_transcriber_notes(tokens):

    if not any(b in line for line in tokens for b in ["{", "}"]):
        return tokens

    new_tokens = []

    for line in tokens:
        note_starts, note_ends = [], []
        for i, token in enumerate(line):
            if "{" in token:
                note_starts.append(i)
            elif "}" in token:
                note_ends.append(i)

        assert(len(note_starts) == len(note_ends))
        assert(all([note_starts[i] <= note_ends[i] for i in range(len(note_starts))]))
        excluded_indices = []
        for s,e in zip(note_starts, note_ends):
            excluded_indices.extend(range(s,e+1))
        new_line = []
        for i,x in enumerate(line):
            if i not in excluded_indices:
                new_line.append(x)
        if new_line:
            new_tokens.append(new_line)

    return new_tokens