import os
import json
from utils import untokenize, convert_REC_ID_to_DD_ID, get_DD_by_ID
from tqdm import tqdm

emo_dict = {'0': 'neutral', '1': 'anger', '2': 'disgust', '3': 'fear', '4': 'happiness', '5': 'sadness', '6': 'surprise'}


# Load original DailyDialog data
DD_data = json.load(open('TLiDB_DailyDialog/TLiDB_DailyDialog.json', 'r'))

# Load RECCON specific data
RECCON_train = json.load(open('RECCON_train.json', 'r'))
RECCON_validation = json.load(open('RECCON_validation.json', 'r'))
RECCON_test = json.load(open('RECCON_test.json', 'r'))
RECCON_data = RECCON_train
RECCON_data.update(RECCON_validation)
RECCON_data.update(RECCON_test)

def untokenize_REC_dialogue(dialogue):
    for turn in dialogue:
        turn['utterance'] = untokenize(turn['utterance'].split())

def update_annotations_from_RECCON(DD_data, RECCON_data):

    num_emotion_updates = 0
    for REC_ID, REC_dialogue in tqdm(RECCON_data.items()):
        DD_ID = convert_REC_ID_to_DD_ID(REC_ID)
        DD_datum = get_DD_by_ID(DD_ID, DD_data)

        untokenize_REC_dialogue(REC_dialogue[0])

        # fix typos, errors, etc.
        if DD_ID == "dialogue-3186":
            DD_datum['dialogue'][11]['utterance'] = "Yes, and I took many pictures."
        if DD_ID == "dialogue-3072":
            DD_datum['dialogue'][3]['utterance'] = "Don't judge a book by its cover. Do you know what it's about?"
        if DD_ID == "dialogue-12401":
            DD_datum['dialogue'][2]['utterance'] = "Then you will be happy to hear that today all our pizzas are on sale. Two for one."
        if DD_ID == "dialogue-12923":
            DD_datum['dialogue'][10]['utterance'] = REC_dialogue[0][10]['utterance']
            for i in range(11,16):
                DD_datum['dialogue'][i]['utterance'] = REC_dialogue[0][i]['utterance']
                DD_datum['dialogue'][i]['emotion_recognition'] = DD_datum['dialogue'][i+1]['emotion_recognition']
                DD_datum['dialogue'][i]['dialogue_actions'] = DD_datum['dialogue'][i+1]['dialogue_actions']
            DD_datum['dialogue'] = DD_datum['dialogue'][:-1]
        if DD_ID == "dialogue-3814":
            DD_datum['dialogue'][4]['utterance'] = "Here we go. There is a museum of the Beijing Opera art."


        # make sure that the dialogue is identical between DD and RECCON
        # accept updated emotions from RECCON
        for i in range(len(REC_dialogue[0])):
            if REC_dialogue[0][i]['utterance'] != DD_datum['dialogue'][i]['utterance']:
                print(REC_dialogue[0][i]['utterance'])
                print(DD_datum['dialogue'][i]['utterance'])
                print(len(REC_dialogue[0][i]['utterance']))
                print(len(DD_datum['dialogue'][i]['utterance']))
                print(DD_ID)
                print(REC_ID)
                assert(REC_dialogue[0][i]['utterance'] == DD_datum['dialogue'][i]['utterance'])
            if REC_dialogue[0][i]['emotion'] != DD_datum['dialogue'][i]['emotion_recognition']:
                print(f"Updated \"{REC_dialogue[0][i]['utterance']}\" from {DD_datum['dialogue'][i]['emotion_recognition']} to {REC_dialogue[0][i]['emotion']}")
                # fix RECCON typo
                if REC_dialogue[0][i]['emotion'] in ['happy','happines', "excited"]:
                    REC_dialogue[0][i]['emotion'] = 'happiness'
                if REC_dialogue[0][i]['emotion'] == 'angry':
                    REC_dialogue[0][i]['emotion'] = 'anger'
                if REC_dialogue[0][i]['emotion'] == 'sad':
                    REC_dialogue[0][i]['emotion'] = 'sadness'
                if REC_dialogue[0][i]['emotion'] == 'surprised':
                    REC_dialogue[0][i]['emotion'] = 'surprise'
                DD_datum['dialogue'][i]['emotion_recognition'] = REC_dialogue[0][i]['emotion']
                num_emotion_updates += 1

    print(f"Updated emotions on {num_emotion_updates} utterances")
    return DD_data

def update_DD_turn_utterance(DD_datum, turn_num, new_utterance):
    for turn in DD_datum['dialogue']:
        if turn['turn'] == turn_num:
            turn['utterance'] = new_utterance

def update_annotations_from_DDpp(DD_data):

    # add improved annotations from DailyDialogue++
    # mostly typos
    for datum in DD_data['data']:
        if datum['dialogue_id'] in ["dialogue-1566","dialogue-2113", "dialogue-2186", "dialogue-2260", "dialogue-3659", "dialogue-3770"]:
            for turn in datum['dialogue']:
                if "Mr. : " in turn['utterance']:
                    turn['utterance'] = turn['utterance'].replace("Mr. : ", "")
        if datum['dialogue_id'] == "dialogue-6445":
            for turn in datum['dialogue']:
                if turn['turn_id'] == '6':
                    turn['utterance'] = "OK. Then I want to ask you some questions about the tourist guide. If there was an accident, for example a tourist falls ill, what would you do?"
        if datum['dialogue_id'] == "dialogue-9569":
            for turn in datum['dialogue']:
                if turn['turn_id'] == '5':
                    turn['utterance'] = "Yes, I can. You know Mr. Macready, the famous car dealer here in New York. He agreed to act as my guarantor of a loan in the sum of US $10,000 until the end of this year."
        if datum['dialogue_id'] == "dialogue-363":
            for turn in datum['dialogue']:
                if turn['turn_id'] == '1':
                    turn['utterance'] = "Lodge, You must come around sometime, You have to check out my new stereo."
        if datum['dialogue_id'] == "dialogue-10059":
            for turn in datum['dialogue']:
                if turn['turn_id'] == '1':
                    turn['utterance'] = "Do you feel alright?"
        if datum['dialogue_id'] == "dialogue-3081":
            for turn in datum['dialogue']:
                if turn['turn_id'] == '3':
                    turn['utterance'] = "Poor John! He is sandwiched between them."
        if datum['dialogue_id'] == 'dialogue-3702':
            for turn in datum['dialogue']:
                if turn['turn_id'] == '3':
                    turn['utterance'] = "Poor John! He is sandwiched between both of them."
        if datum['dialogue_id'] == "dialogue-3663":
            for turn in datum['dialogue']:
                if turn['turn_id'] == '4':
                    turn['utterance'] = "He certainly is. He is the Beckham of our college."
        if datum['dialogue_id'] == "dialogue-225":
            for turn in datum['dialogue']:
                if turn['turn_id'] == '1':
                    turn['utterance'] = "I think the biggest environmental problem in my country is air pollution."
        if datum['dialogue_id'] == "dialogue-51":
            for turn in datum['dialogue']:
                if turn['turn_id'] == '4':
                    turn['utterance'] = "I wish the store close to us was open 24 - hours a day."
        if datum['dialogue_id'] == "dialogue-376":
            for turn in datum['dialogue']:
                if turn['turn_id'] == '2':
                    turn['utterance'] = "Hi, Lin Tao. I haven't seen you for some time."
        if datum['dialogue_id'] == "dialogue-305":
            for turn in datum['dialogue']:
                if turn['turn_id'] == '2':
                    turn['utterance'] = "That's a great idea. It's the best season for tourists in Florida. You can also get a good suntan there."
        if datum['dialogue_id'] == "dialogue-7":
            for turn in datum['dialogue']:
                if turn['turn_id'] == '4':
                    turn['utterance'] = "Who is he marrying?"
        if datum['dialogue_id'] == "dialogue-3":
            for turn in datum['dialogue']:
                if turn['turn_id'] == '4':
                    turn['utterance'] = "The radio has too many commercials."
        if datum['dialogue_id'] == "dialogue-5112":
            for turn in datum['dialogue']:
                if turn['turn_id'] == '2':
                    turn['utterance'] = "Yes, could you cash these traveler's check for me, please?"
        if datum['dialogue_id'] in ["dialogue-2668",'dialogue-4052']:
            for turn in datum['dialogue']:
                if turn['turn_id'] == '1':
                    turn['utterance'] = "You are blue in the face, aren't you?"
        if datum['dialogue_id'] == "dialogue-347":
            for turn in datum['dialogue']:
                if turn['turn_id'] == '5':
                    turn['utterance'] = "See you on Monday. Have a great weekend."
        if datum['dialogue_id'] == "dialogue-7944":
            for turn in datum['dialogue']:
                if turn['turn_id'] == '7':
                    turn['utterance'] = "How about 10% ? This price is already a little tight, our profit margin is not that large."
        if datum['dialogue_id'] == "dialogue-377":
            for turn in datum['dialogue']:
                if turn['turn_id'] == '2':
                    turn['utterance'] = "Yeah, he has a good command of computer skills. But. . ."
        if datum['dialogue_id'] == "dialogue-1549":
            for turn in datum['dialogue']:
                if turn['turn_id'] == '5':
                    turn['utterance'] = "And why are you yawning now? Are you bored?"
        if datum['dialogue_id'] == "dialogue-139":
            for turn in datum['dialogue']:
                if turn['turn_id'] == '3':
                    turn['utterance'] = "Time to order. Barista, today I want a skinny triple latte."
        if datum['dialogue_id'] == "dialogue-1172":
            for turn in datum['dialogue']:
                if turn['turn_id'] == '2':
                    turn['utterance'] = "I'm planning to go bowling tonight. Tomorrow I was hoping to see a movie. What about you?"
        if datum['dialogue_id'] == "dialogue-4":
            for turn in datum['dialogue']:
                if turn['turn_id'] == '2':
                    turn['utterance'] = "I will be alright soon. I was terrified when I watched them fall from the wire."
        if datum['dialogue_id'] == "dialogue-91":
            for turn in datum['dialogue']:
                if turn['turn_id'] == '2':
                    turn['utterance'] = "Sorry, I'm engaged for the foxtrot. Will the next waltz be alright?"
        if datum['dialogue_id'] == "dialogue-6555":
            for turn in datum['dialogue']:
                if turn['turn_id'] == '5':
                    turn['utterance'] = "Can it go in an envelope?"
        if datum['dialogue_id'] == "dialogue-5160":
            for turn in datum['dialogue']:
                if turn['turn_id'] == '5':
                    turn['utterance'] = "OK. Can I have your original ID card and two 2 - inch photos?"
        if datum['dialogue_id'] in ["dialogue-90","dialogue-10556"]:
            for turn in datum['dialogue']:
                if turn['turn_id'] == '1':
                    turn['utterance'] = "You've been work here for nearly a month, how do you feel about the job?"
        if datum['dialogue_id'] == "dialogue-487":
            for turn in datum['dialogue']:
                if turn['turn_id'] == '3':
                    turn['utterance'] = "He can repair cars? I cannot believe it."
        if datum['dialogue_id'] == "dialogue-169":
            for turn in datum['dialogue']:
                if turn['turn_id'] == '1':
                    turn['utterance'] = "Everyone wants to be financial lose kill. You must invest your savings if you planned to retire rich. I'm thinking about buying stocks. It can be a good investment if I can manage well. What do you say?"
        

    return DD_data

DD_data = update_annotations_from_RECCON(DD_data, RECCON_data)
DD_data = update_annotations_from_DDpp(DD_data)

with open('TLiDB_DailyDialog/TLiDB_DailyDialog.json',"w", encoding='utf8') as f:
    json.dump(DD_data, f, indent=2, ensure_ascii=False)