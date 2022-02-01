from .TLiDB_dataset import TLiDB_Dataset, load_split_ids
import random

class Friends_dataset(TLiDB_Dataset):
    """
    Friends dataset
    # TODO: Fill in metadata
    
    """

    _dataset_name = "Friends"
    _tasks = [
        'emotion_recognition', 'reading_comprehension', 'character_identification',
        'question_answering', 'personality_detection', 'relation_extraction',
        'MELD_emotion_recognition', 'response_generation'
    ]
    _url = "https://drive.google.com/uc?export=download&id=1QK_XX-d38fKeJlcTcoMT9ku3VZn6bv6H"
    _task_metadatas = {
        "emotion_recognition": {
                "prompt":"emotion:","type":"classification","loader":"utterance_level_classification",
                "collate_type":"classification"
        },
        "reading_comprehension": {
            "prompt": "Fill in [PLACEHOLDER]:","type":"span_extraction",
            "loader": "character_span_extraction", "collate_type":"character_span_extraction"
        },
        "character_identification": {
            "prompt": "{} in the phrase '{}' refers to ", "type":"classification", "loader": "character_identification",
            "collate_type":"character_identification"
        },
        "question_answering": {
            "prompt":"","type":"span_extraction","loader":"span_extraction",
            "collate_type":"span_extraction"
        },
        "personality_detection": {
            "prompt":"","type":"multioutput_classification","num_outputs":5,"num_labels":2,
            "loader":"personality_detection","collate_type":"personality_detection"
        },
        "relation_extraction": {
            "prompt":"","type":"multilabel_classification","loader":"relation_extraction",
            "collate_type":"relation_extraction" 
        },
        "MELD_emotion_recognition": {
            "prompt":"emotion", "type":"classification", "loader":"utterance_level_classification",
            "collate_type":"classification"
        },
        "response_generation": {
            "prompt": "", "type":"response_generation","loader":"response_generation"
        }
    }
    def __init__(self, task, dataset_folder, model_type, split, few_shot_percent=None):
        assert task in self._tasks, f"{task} is not a valid task for {self._dataset_name}"
        super().__init__(self._dataset_name, task, model_type, dataset_folder=dataset_folder)
        self._task_metadata = self._task_metadatas[task]
        self._input_array = []
        self._y_array = []
        self._metadata_fields = []
        self._metadata_array = []
        split_ids = load_split_ids(self._dataset_name, dataset_folder, split, few_shot_percent)
        self._load_data(task, split_ids)
        self._num_classes = len(self.task_labels)
        self._y_size = len(self._y_array)

    def _load_data(self, task, split_ids):
        # get the data loader, based on whether the task is utterance level/dialogue level/span extraction/etc.
        loader = getattr(self, f"_load_{self._task_metadata['loader']}_task")
        return loader(task,split_ids)

    def _load_utterance_level_classification_task(self, task, split_ids):
        for datum in self.dataset['data']:
            if datum['dialogue_id'] in split_ids:
                if task in datum['dialogue_metadata']:
                    dialogue = []
                    for turn in datum['dialogue']:
                        dialogue.append([" ".join(turn['speakers']), turn['utterance']])
                        str_dialogue = self._convert_dialogue_to_string(dialogue)
                        if task in turn:
                            self._input_array.append(str_dialogue)
                            self._y_array.append(turn[task])

    def _load_character_span_extraction_task(self, task, split_ids):
        for datum in self.dataset['data']:
            if datum['dialogue_id'] in split_ids:
                if task in datum['dialogue_metadata']:
                    dialogue = [[" ".join(turn['speakers']), turn['utterance']] for turn in datum['dialogue']]
                    str_dialogue = self._convert_dialogue_to_string(dialogue)

                    entity_list = datum['dialogue_metadata'][task]['scene_entities']
                    str_entity_list = " or ".join(entity_list)
                    for qa in datum[task]:
                        self._input_array.append({
                            "context": str_dialogue,
                            "question": qa['query'],
                            "entity_list": str_entity_list
                        })
                        answer_start = str_entity_list.find(qa['answer'])
                        assert(answer_start != -1), "couldn't find exact entity name"
                        answer = {
                            "text":qa['answer'],
                            "answer_start":answer_start
                        }
                        self._y_array.append(answer)

    def _load_character_identification_task(self, task, split_ids):
        for datum in self.dataset['data']:
            if datum['dialogue_id'] in split_ids:
                if task in datum['dialogue_metadata']:
                    dialogue = []
                    for turn in datum['dialogue']:
                        dialogue.append([" ".join(turn['speakers']), turn['utterance']])
                        if task in turn:
                            for entity_mention in turn[task]:
                                # add context surrounding the reference
                                post_context_words = turn['utterance'][entity_mention['start']:].split()
                                pre_context_words = turn['utterance'][:entity_mention['start']+len(entity_mention['entity_reference'])+1].split()
                                
                                # prefer whichever context is longer, post context wins in a tie
                                if len(pre_context_words) > len(post_context_words):
                                    context_words = pre_context_words[-4:]
                                else:
                                    context_words = post_context_words[:4]

                                entity_context = " ".join(context_words)

                                self._input_array.append({
                                    "context":dialogue.copy(),
                                    "entity_context": entity_context,
                                    "entity_reference": entity_mention['entity_reference']
                                })
                                self._y_array.append(entity_mention['normalized_entity'])

    def _load_span_extraction_task(self, task, split_ids):
        for datum in self.dataset['data']:
            if datum['dialogue_id'] in split_ids:
                if task in datum['dialogue_metadata']:
                    dialogue = [[" ".join(turn['speakers']), turn['utterance']] for turn in datum['dialogue']]
                    str_dialogue = self._convert_dialogue_to_string(dialogue)
                    for qas in datum[task]:
                        for qa in qas['qas']:
                            answers = []
                            for answer in qa['answers']:
                                prior_turns = self._convert_dialogue_to_string(dialogue[:answer['answer_utterance_id']])
                                
                                # add 1 for space between prior turns and current turn
                                if prior_turns:
                                    answer_start = len(prior_turns) + 1
                                else:
                                    answer_start = 0

                                if not answer['is_speaker']:
                                    if dialogue[answer['answer_utterance_id']][0]:
                                        # if there is a speaker, add speaker + 2 for ': ' between current speaker and current turn
                                        answer_start += len(dialogue[answer['answer_utterance_id']][0]) + 2
                                    answer_start += answer['answer_start']
                                test = str_dialogue[answer_start:answer_start+len(answer['text'])]
                                assert(answer['text'] == test), "answer text doesn't match"
                                answers.append({
                                    "text": answer['text'],
                                    "answer_start": answer_start + len(qa['question']) + 1
                                })
                            self._input_array.append({
                                "context": str_dialogue, "question": qa['question']
                            })
                            self._y_array.append(answers)
                            
    def _load_personality_detection_task(self, task, split_ids):
        for datum in self.dataset['data']:
            if datum['dialogue_id'] in split_ids:
                if task in datum['dialogue_metadata']:
                    for sample in datum[task]:
                        dialogue = [[" ".join(datum['dialogue'][i]['speakers']), datum['dialogue'][i]['utterance']] for i in sample['turns']]
                        str_dialogue = self._convert_dialogue_to_string(dialogue)
                        self._input_array.append({
                            "context": str_dialogue,
                            "focus_speaker": sample['focus_speaker']
                        })
                        self._y_array.append({
                            "personality_characteristics":sample['personality_characteristics'],
                            "labels": list(sample['personality_characteristics'].values())
                        })


    def _load_relation_extraction_task(self, task, split_ids):
        for datum in self.dataset['data']:
            if datum['dialogue_id'] in split_ids:
                if task in datum['dialogue_metadata']:
                    for sample in datum[task]:
                        dialogue = [[" ".join(datum['dialogue'][i]['speakers']), datum['dialogue'][i]['utterance']] for i in sample['turns']]
                        str_dialogue = self._convert_dialogue_to_string(dialogue)
                        for triple in sample['relation_triples']:
                            self._input_array.append({
                                "context": str_dialogue,
                                "head":triple['head'],
                                "tail":triple['tail']
                            })
                            self._y_array.append(triple['relations'])


    def _collate_encoder(self, batch):
        X, y, metadata = [], [], {}
        for item in batch:
            if self._task_metadata['collate_type'] == "classification":
                X.append(item[0])
            elif self._task_metadata['collate_type'] == "character_span_extraction":
                X.append(self._join_strings(item[0]['entity_list'],"[SEP]",item[0]['question'],"[SEP]",item[0]['context']))
            elif self._task_metadata['collate_type'] == "character_identification":
                context = self._convert_dialogue_to_string(item[0]['context'][::-1])
                prompt = f"Who does '{item[0]['entity_reference']}' refer to in the phrase '{item[0]['entity_context']}'?"
                X.append(self._join_strings(prompt, "[SEP]", context))
            elif self._task_metadata['collate_type'] == "span_extraction":
                X.append(self._join_strings(item[0]['question'],item[0]['context']))
            elif self._task_metadata['collate_type'] == "personality_detection":
                X.append(self._join_strings(item[0]['focus_speaker'],"[SEP]",item[0]['context']))
            elif self._task_metadata['collate_type'] == "relation_extraction":
                X.append(self._join_strings(item[0]['head'],"[SEP]",item[0]['tail'],"[SEP]",item[0]['context']))
            else:
                raise NotImplementedError(f"Collate type {self._task_metadata['collate_type']} not implemented")
            y.append(item[1])
            for k,v in item[2].items():
                if k not in metadata:
                    metadata.append(k)
                metadata[k].append(v)
        return X, y, metadata