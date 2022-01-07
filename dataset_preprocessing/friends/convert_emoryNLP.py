# some code taken from Jinho D. Choi at https://github.com/emorynlp/character-mining

import os
import json
import glob
import re

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

emo_dict = {"0": 'Joyful', "1": 'Mad',"2": 'Neutral', "3": 'Peaceful',
            "4": 'Powerful', "5": 'Sad', "6": 'Scared'}

def untokenize(words):
    """
    Untokenizing a text undoes the tokenizing operation, restoring
    punctuation and spaces to the places that people expect them to be.
    Ideally, `untokenize(tokenize(text))` should be identical to `text`,
    except for line breaks.
    """
    text = ' '.join(words)
    step1 = text.replace("。",".").replace("’","'").replace("`` ", '"').replace(" ''", '"').replace('. . .', '...').replace(" ` ", " '").replace(" ,",",")
    step2 = step1.replace("( ","(").replace(" ( ", " (").replace(" )", ")").replace(" ) ", ") ").replace(" -- ", " - ").replace("—","-").replace("–","-").replace('”','"').replace('“','"').replace("‘","'").replace("’","'")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = re.sub(r'(?<=[.,])(?=[^\s])', r' ', step4)
    step6 = step5.replace(" '", "'").replace(" n't", "n't").replace("n' t", "n't").replace("t' s","t's").replace("' ll", "'ll").replace("I' m", "I'm").replace(
        "can not", "cannot").replace("I' d", "I'd").replace("' re", "'re").replace("t ' s", "t's").replace("e' s", "e's")
    step7 = re.sub(r'\$\s(\d)', r'$\1', step6)
    step8 = re.sub(r'(\d),\s?(\d\d\d)', r'\1,\2', step7)
    step9 = re.sub(r'(\d.) (\d\%)', r'\1\2', step8)
    step10 = step9.replace("? !", "?!").replace("! !", "!!").replace("! ?", "!?").replace("n'y","n't").replace('yarning','yawning').replace(" om V", " on V")
    return step10.strip()

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

def add_character_identification(formatted_data):
    def get_entities(entity_list):
        return [entity for entity in entity_list if entity[-1] != "Non-Entity"]

    pass

def add_emotion_detection(formatted_data):
    pass

def add_reading_comprehension(formatted_data):
    pass

def convert_season_dialogues(season_raw, formatted_data):
    # For each dialogue, format the dialogue with speakers and a turn id

    for episode in season_raw[EPISODES]:
        for scene in episode[SCENES]:
            scene_id = scene[SCENE_ID]
            formatted_datum = {
                "dialogue_id": scene_id,
                "dialogue_metadata": {},
                "dialogue": [],
            }
            turn_id = 0
            for utterance in scene[UTTERANCES]:
                speakers = utterance[SPEAKERS]
                tokens = utterance[TOKENS_WITH_NOTE] if utterance[TOKENS_WITH_NOTE] else utterance[TOKENS]
                utt = ' '.join([untokenize(u) for u in tokens])

                formatted_turn = {
                    "turn_id": turn_id,
                    "speaker": speakers,
                    "utterance": utt,
                }
                formatted_datum["dialogue"].append(formatted_turn)
                turn_id += 1
            formatted_data['data'].append(formatted_datum)


emoryNLP_dir = "emoryNLP"
general_stats(emoryNLP_dir)

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
                "metric_kwargs": {"f1": [{"average": "micro"}, {"average": "macro"}]}
            },
            "reading_comprehension": { # formulated as span extraction? for training, but evaluated as accuracy
                "metrics": ["accuracy"],
            },
            "character_identification": {
                "labels": [],
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