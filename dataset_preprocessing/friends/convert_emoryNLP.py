# some code taken from Jinho D. Choi at https://github.com/emorynlp/character-mining

import os
import json
import glob
from tqdm import tqdm
from utils import untokenize, remove_notes_from_utt, remove_transcriber_notes, OPENERS, CLOSERS

SEASON_ID = 'season_id'
EPISODES = 'episodes'
EPISODE_ID = 'episode_id'
EPISODE = 'episode'
SCENES = 'scenes'
SCENE_ID = 'scene_id'
UTTERANCES = 'utterances'
UTTERANCE_ID = 'utterance_id'
SPEAKERS = 'speakers'
TRANSCRIPT = 'transcript'
TRANSCRIPT_WITH_NOTE = 'transcript_with_note'
TOKENS = 'tokens'
TOKENS_WITH_NOTE = 'tokens_with_note'

# reading comprehension
RC_ENTITIES = 'rc_entities'
PLOTS = 'plots'
P_ENT = 'p_ent'
U_ENT = 'u_ent'
S_ENT = 's_ent'
QUERY = 'query'
ANSWER = 'answer'
PLACEHOLDER = "[PLACEHOLDER]"


emo_dict = {"0": 'Joyful', "1": 'Mad',"2": 'Neutral', "3": 'Peaceful',
            "4": 'Powerful', "5": 'Sad', "6": 'Scared'}

character_mapping = {
    "Chandler Bing":"chandler",
    "Joey Tribbiani":"joey",
    "Monica Geller":"monica",
    "Phoebe Buffay":"phoebe",
    "Rachel Green":"rachel",
    "Ross Geller":"ross"
    }

def general_stats(json_dir):
    def stats(json_file):
        num_scenes = 0
        num_utterances = 0
        num_utterances_wn = 0
        num_sentences = 0
        num_sentences_wn = 0
        num_tokens = 0
        num_tokens_wn = 0
        speaker_list = set()

        season = json.load(open(json_file))
        episodes = season[EPISODES]

        for episode in episodes:
            scenes = episode[SCENES]
            num_scenes += len(scenes)

            for scene in scenes:
                utterances = scene[UTTERANCES]
                num_utterances_wn += len(utterances)

                for utterance in utterances:
                    speaker_list.update(utterance[SPEAKERS])

                    tokens = utterance[TOKENS]
                    if tokens:
                        num_utterances += 1
                        num_sentences += len(tokens)
                        num_tokens += sum([len(t) for t in tokens])

                    tokens_wn = utterance[TOKENS_WITH_NOTE] or tokens
                    num_sentences_wn += len(tokens_wn)
                    num_tokens_wn += sum([len(t) for t in tokens_wn])

        return [season[SEASON_ID], len(episodes), num_scenes, num_utterances, num_sentences, num_tokens, speaker_list,
                num_utterances_wn, num_sentences_wn, num_tokens_wn]

    g_speaker_list = set()
    print('\t'.join(['Season ID', 'Episodes', 'Scenes', 'Utterances', 'Sentences', 'Tokens', 'Speakers', 'Utterances (WN)', 'Sentences (WN)', 'Tokens (WN)']))
    for json_file in sorted(glob.glob(os.path.join(json_dir, '*.json'))):
        l = stats(json_file)
        g_speaker_list.update(l[6])
        l[6] = len(l[6])
        print('\t'.join(map(str, l)))
    print('All speakers: %s' % (len(g_speaker_list)))

def is_sublist(l1, l2):
    # check if l1 is a sublist of l2 after removing notes from l2
    l2_no_note = remove_notes_from_utt(l2)
    if l1 == l2_no_note:
        return True

    return False

def shift_entity_indices_for_notes(entities, tokens_with_notes):
    if not entities:
        return
    note_starts = [i for i, x in enumerate(tokens_with_notes) if any(opener in x for opener in OPENERS)]
    note_ends = [i for i, x in enumerate(tokens_with_notes) if any(closer in x for closer in CLOSERS)]
    if note_starts:
        assert(len(note_starts) == len(note_ends))
        assert(
            all([note_starts[i] < note_ends[i] for i in range(len(note_starts))]) and \
            all([note_starts[i+1] > note_ends[i] for i in range(0, len(note_starts)-1)])
            )

        for s, e in zip(note_starts, note_ends):
            note_length = e - s + 1
            for entity in entities:
                if entity[0] >= s:
                    entity[0] += note_length
                    entity[1] += note_length

def unify_tokens_entities_with_notes(tokens, tokens_with_notes, entities):

    for i in range(len(tokens_with_notes)):

        if i >= len(tokens):
            tokens.append([])
            entities.append([])
        elif tokens_with_notes[i] != tokens[i]:

            # if utterance has a note, shift entity indices
            if is_sublist(tokens[i], tokens_with_notes[i]):
                shift_entity_indices_for_notes(entities[i], tokens_with_notes[i])

            # if utterance is only a note, add empty entries for tokens, entities
            else:
                tokens.insert(i, [])
                entities.insert(i, [])

    return tokens, entities

def get_entities(entity_list, tokens, tokens_with_note):
    if tokens_with_note and tokens != tokens_with_note:
        tokens, entity_list = unify_tokens_entities_with_notes(tokens, tokens_with_note, entity_list)

    utterance_list = tokens_with_note or tokens
    entities = []
    utt_len = 0
    for e_list, u_list in zip(entity_list, utterance_list):
        utt = untokenize(u_list)
        if e_list:
            # track entity references so we don't duplicate them, 
            #    e.g. if we have 'I' multiple times in an utterance
            seen_entities = {}

            for e in e_list:
                # get the original string for this entity
                ent_reference = untokenize(u_list[e[0]:e[1]])

                # if we've already seen this entity, find it starting from the previously seen index
                if ent_reference in seen_entities:
                    # start = utt.find(ent_reference, seen_entities[ent_reference])+utt_len
                    in_utt_start = utt.find(ent_reference, seen_entities[ent_reference])
                else:
                    # start = utt.find(ent_reference)+utt_len
                    in_utt_start = utt.find(ent_reference)

                global_start = in_utt_start + utt_len
                
                # ensure that the entity in the original list, and the created string are the same
                found_ent = ' '.join([untokenize(u) for u in utterance_list])[global_start:global_start+len(ent_reference)]
                if found_ent != ent_reference:
                    full = ' '.join([untokenize(u) for u in utterance_list])
                    found_ent = ' '.join([untokenize(u) for u in utterance_list])[global_start:global_start+len(ent_reference)]

                normalized_entity = character_mapping[e[2]] if e[2] in character_mapping else "other"
                entities.append({
                    'entity': e[2],
                    'normalized_entity': normalized_entity,
                    'start': global_start,
                    'entity_reference': ent_reference,
                })

                seen_entities[ent_reference] = in_utt_start + len(ent_reference)

        utt_len += len(utt) + 1

    return entities

def get_reading_comprehension_annotations(plots, dialogue_entities):
    annotations = []
    entity_set = set()
    for i, passage in enumerate(plots):
        passage_entities = {}
        for ent, index_list in dialogue_entities.items():
            for index in index_list[P_ENT]:
                if i == index[0]:
                    if ent not in passage_entities:
                        passage_entities[ent] = []
                    passage_entities[ent].append([index[1], index[2]])
        
        for ent in passage_entities:
            for ent_position in passage_entities[ent]:
                ent_start = ent_position[0]
                ent_end = ent_position[1]
                masked_passage = []
                masked = False
                for j, token in enumerate(passage.split(" ")):
                    if j >= ent_start and j < ent_end:
                        if not masked:
                            masked_passage.append(PLACEHOLDER)
                            masked = True
                        continue
                    else:
                        masked_passage.append(token)
                
                annotations.append({
                    "query": untokenize(masked_passage),
                    "answer": ent
                })
                entity_set.add(ent)

    return annotations, entity_set

def combine_notes(tokens_with_notes):
    open = 0
    open_utterances = []
    for i in range(len(tokens_with_notes)):
        for t in tokens_with_notes[i]:
            if any(opener in t for opener in OPENERS):
                open += 1
            if any(closer in t for closer in CLOSERS):
                open -= 1
        if open > 0:
            open_utterances.append(i)

    # for each open note, add the next note
    for i in open_utterances[::-1]:
        try:
            tokens_with_notes[i].extend(tokens_with_notes[i+1])
            del tokens_with_notes[i+1]
        except IndexError:
            # some notes have an opener without a closer
            if not any(closer in tokens_with_notes[i][-1] for closer in CLOSERS):
                closed = False
                for j, opener in enumerate(OPENERS):
                    if opener in tokens_with_notes[i]:
                        tokens_with_notes[i].append(CLOSERS[j])
                        closed = True
                        break
                assert(closed), "Unable to resolve unbalanced parentheses"

def convert_season_dialogues(season_raw, formatted_data):
    # For each dialogue, format the dialogue with speakers and a turn id
    for episode in tqdm(season_raw[EPISODES], desc = season_raw[SEASON_ID]):
        for scene in episode[SCENES]:
            scene_id = scene[SCENE_ID]
            formatted_datum = {
                "dialogue_id": scene_id,
                "dialogue_metadata": {},
                "dialogue": [],
            }
            turn_id = 0

            # Add formatted turn along with turn-level task annotations
            for utterance in scene[UTTERANCES]:

                # fix specific utterances
                if utterance['utterance_id'] == "s01_e14_c08_u004":
                    utterance[TOKENS_WITH_NOTE][2][0] = ")"
                if utterance['utterance_id'] == "s02_e14_c01_u004":
                    utterance[TOKENS][2].append("]")
                if utterance['utterance_id'] == "s04_e03_c07_u005":
                    utterance[TOKENS_WITH_NOTE][0].append(")")
                if utterance['utterance_id'] == "s04_e23_c02_u038":
                    utterance[TOKENS_WITH_NOTE][0].append(")")
                if utterance['utterance_id'] == "s05_e14_c09_u001":
                    utterance[TOKENS_WITH_NOTE][1].append("]")
                if utterance['utterance_id'] == "s05_e21_c09_u001":
                    utterance[TOKENS_WITH_NOTE][6].append("}")
                    utterance[TOKENS_WITH_NOTE][7] = ["]"]
                if utterance['utterance_id'] == "s05_e24_c14_u001":
                    utterance[TOKENS_WITH_NOTE] = [['[', 'Scene:', 'A', 'blackjack', 'table,', 'Joey', 'is', 'moving', 'in', 'to', 'try', 'and', 'get', 'his', 'hand', 'twin', ']']]
                if utterance['utterance_id'] == "s06_e14_c06_u037":
                    utterance[TOKENS_WITH_NOTE][1].append(")")
                if utterance['utterance_id'] == "s06_e14_c08_u015":
                    utterance[TOKENS_WITH_NOTE][1].append(")")
                if utterance['utterance_id'] == "s06_e20_c09_u036":
                    utterance[TOKENS_WITH_NOTE][1] = ['Monica', 'removes', 'Rachel', "'s", 'sock', 'and', 'starts', 'beating', 'her', 'with', 'it', '.']
                if utterance['utterance_id'] == "s06_e23_c04_u003":
                    utterance[TOKENS][7].append(")")
                if utterance['utterance_id'] == "s07_e21_c05_u037":
                    utterance[TOKENS_WITH_NOTE][1].append(")")
                if utterance['utterance_id'] == "s08_e03_c12_u012":
                    utterance[TOKENS][1].append(")")
                if utterance['utterance_id'] == "s08_e07_c13_u004":
                    utterance[TOKENS][1].append(")")
                if utterance['utterance_id'] == "s09_e09_c10_u005":
                    utterance[TOKENS_WITH_NOTE][5] = [")"]
                if utterance['utterance_id'] == "s10_e07_c12_u000":
                    utterance[TOKENS_WITH_NOTE] = [['[', 'Scene:', 'The', 'New', 'York', 'City', "Children's", 'Fund', 'building.', 'Phoebe', 'and', 'Mike', 'are', 'entering.', ']']]
                if utterance['utterance_id'] == "s10_e08_c08_u007":
                    utterance[TOKENS_WITH_NOTE][2][-1] = ")"
                if utterance['utterance_id'] == "s10_e09_c01_u021":
                    utterance[TOKENS_WITH_NOTE][-1][0] = "("
                if utterance['utterance_id'] == "s10_e13_c09_u016":
                    utterance[TOKENS_WITH_NOTE][1].insert(0, "(")

                # combine utterance that was split into 2 lines
                if utterance['utterance_id'] == 's05_e02_c01_u020':
                    utterance[TOKENS_WITH_NOTE] = [['Yeah,', 'can', 'I', 'get', 'a', '3', '-', 'piece,', 'some', 'cole', 'slaw,', 'some', 'beans,', 'and', 'a', 'Coke', '-', '(', 'Yelps', 'in', 'pain', 'as', 'Monica', 'grabs', 'him', 'underwater', ')', '-', 'Diet', 'Coke.']]
                if utterance['utterance_id'] == 's05_e02_c01_u021':
                    continue

                # replace the format: "\d )" with "\d "
                if utterance['utterance_id'] == 's05_e13_c11_u012':
                    utterance[TOKENS_WITH_NOTE] = [['Okay.', 'I', 'have', 'just', 'a', 'few', 'questions', 'to', 'ask', 'so', "I'm", 'going', 'to', 'get', 'out', 'my', 'official', 'forms.', '(', 'She', 'picks', 'up', 'a', 'couple', 'of', 'crumpled', 'receipts.', ')', 'Okay,', 'so,', 'question', '1', 'You', 'and', 'uh,', 'you', 'were', 'married', 'to', "Francis'", 'daughter', 'Lilly,', 'is', 'that', 'correct?']]
                if utterance['utterance_id'] == 's05_e13_c11_u014':
                    utterance[TOKENS] = [['Okay,', 'umm,', 'question', '2', 'Umm,', 'did', 'that', 'marriage', 'end', 'A.', 'Happily,', 'B.', 'Medium,', 'or', 'C.', 'In', 'the', 'total', 'abandonment', 'of', 'her', 'and', 'her', 'two', 'children?']]

                # whoever transcribed seasons 5-10 added their own thoughts directly in the transcript
                # remove these
                if utterance['utterance_id'] in ['s05_e17_c10_u021','s05_e21_c01_u017',
                                                's05_e21_c02_u038', 's05_e22_c03_u006',
                                                's05_e24_c09_u007', 's08_e09_c02_u026']:
                    utterance[TOKENS_WITH_NOTE] = utterance[TOKENS]

                # combine notes that were broken into multiple lines
                if utterance[TOKENS_WITH_NOTE]:
                    combine_notes(utterance[TOKENS_WITH_NOTE])

                speakers = utterance[SPEAKERS]
                tokens = utterance[TOKENS_WITH_NOTE] if utterance[TOKENS_WITH_NOTE] else utterance[TOKENS]

                tokens = remove_transcriber_notes(tokens)
                if not tokens:
                    continue

                utt = ' '.join([untokenize(u) for u in tokens])

                formatted_turn = {
                    "turn_id": turn_id,
                    "speakers": speakers,
                    "utterance": utt,
                }

                # get emotion annotations
                if utterance[TOKENS] and 'emotion' in utterance:
                    formatted_turn['emotion_recognition'] = utterance['emotion'][0]
                    if "emotion_recognition" not in formatted_datum['dialogue_metadata']:
                        formatted_datum['dialogue_metadata']['emotion_recognition'] = None

                # get character identification annotations
                if utterance[TOKENS] and 'character_entities' in utterance:
                    formatted_turn['character_identification'] = get_entities(utterance['character_entities'], utterance[TOKENS], utterance[TOKENS_WITH_NOTE])
                    if "character_identification" not in formatted_datum['dialogue_metadata']:
                        formatted_datum['dialogue_metadata']['character_identification'] = None

                formatted_datum["dialogue"].append(formatted_turn)
                turn_id += 1

            # add dialogue level task annotations
            if PLOTS in scene and scene[PLOTS]:
                formatted_datum['reading_comprehension'], scene_entities = get_reading_comprehension_annotations(scene[PLOTS], scene[RC_ENTITIES])
                formatted_datum['dialogue_metadata']['reading_comprehension'] = {"scene_entities":list(scene_entities)}

            formatted_data['data'].append(formatted_datum)

emoryNLP_dir = "emoryNLP"
general_stats(emoryNLP_dir)

TLiDB_path="TLiDB_Friends"
if not os.path.isdir(TLiDB_path):
    os.mkdir(TLiDB_path)

formatted_data = {
    "metadata": {
        "dataset_name": "Friends",
        "tasks": [
            "emotion_recognition",
            "reading_comprehension",
            "character_identification"
        ],
        "task_metadata": {
            "emotion_recognition": {
                "labels": list(emo_dict.values()), "metrics": ["f1"],
                "metric_kwargs": {"f1": [{"average": "micro"}, {"average": "weighted"}]}
            },
            "reading_comprehension": { # formulated as span extraction, but evaluated as accuracy by
                                        #   evaluating only on exact match
                "metrics": ["exact_match"],
            },
            "character_identification": {
                "labels": ["chandler","joey","monica","phoebe","rachel","ross","other"],
                "metrics": ["f1"],
                "metric_kwargs": {"f1": [{"average": "micro"}, {"average": "macro"}]}
            }
        }
    },
    "data": []
}

for json_file in sorted(glob.glob(os.path.join(emoryNLP_dir, '*.json'))):
    season_raw = json.load(open(json_file))
    convert_season_dialogues(season_raw, formatted_data)

with open(os.path.join(TLiDB_path, 'TLiDB_Friends.json'), 'w') as f:
    json.dump(formatted_data, f, indent=2)