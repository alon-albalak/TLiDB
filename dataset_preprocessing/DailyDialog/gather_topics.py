with open("ijcnlp_dailydialog/dialogues_act.txt", "r") as f:
    all_dialog_act = f.readlines()
with open("ijcnlp_dailydialog/dialogues_emotion.txt", "r") as f:
    all_emotion = f.readlines()
with open("ijcnlp_dailydialog/dialogues_topic.txt", "r") as f:
    all_topic = f.readlines()
with open("ijcnlp_dailydialog/dialogues_text.txt", "r") as f:
    all_text = f.readlines()

with open("test/dialogues_test.txt", "r") as f:
    test_text = f.readlines()
with open("test/dialogues_act_test.txt", "r") as f:
    test_act = f.readlines()
with open("test/dialogues_emotion_test.txt", "r") as f:
    test_emotion = f.readlines()

with open("validation/dialogues_validation.txt", "r") as f:
    validation_text = f.readlines()
with open("validation/dialogues_act_validation.txt", "r") as f:
    validation_act = f.readlines()
with open("validation/dialogues_emotion_validation.txt", "r") as f:
    validation_emotion = f.readlines()

with open("train/dialogues_train.txt", "r") as f:
    train_text = f.readlines()
with open("train/dialogues_act_train.txt", "r") as f:
    train_act = f.readlines()
with open("train/dialogues_emotion_train.txt", "r") as f:
    train_emotion = f.readlines()

test_topic, validation_topic, train_topic = [], [], []
non_matching = 0
for i, text in enumerate(test_text):
    all_ind = all_text.index(text)
    if not (
        all_dialog_act[all_ind] == test_act[i] and \
        all_emotion[all_ind] == test_emotion[i]
    ):
        non_matching += 1
    test_topic.append(all_topic[all_ind])

for i, text in enumerate(validation_text):
    all_ind = all_text.index(text)
    if not (
        all_dialog_act[all_ind] == validation_act[i] and \
        all_emotion[all_ind] == validation_emotion[i]
    ):
        non_matching += 1
    validation_topic.append(all_topic[all_ind])

for i, text in enumerate(train_text):
    # a single dialogue has a slight difference between the train and full data files
    try:
        all_ind = all_text.index(text)
    except:
        for j, t in enumerate(all_text):
            if text[:100] == t[:100]:
                all_ind = all_text.index(t)
                break
    if not (
        all_dialog_act[all_ind] == train_act[i] and \
        all_emotion[all_ind] == train_emotion[i]
    ):
        non_matching += 1
    train_topic.append(all_topic[all_ind])


with open("test/dialogues_topic_test.txt", "w") as f:
    for topic in test_topic:
        f.write(topic)
with open("validation/dialogues_topic_validation.txt", "w") as f:
    for topic in validation_topic:
        f.write(topic)
with open("train/dialogues_topic_train.txt", "w") as f:
    for topic in train_topic:
        f.write(topic)
