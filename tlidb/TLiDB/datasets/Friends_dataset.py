from .TLiDB_dataset import TLiDB_Dataset, load_split_ids

relations_to_natural_language_map = {
    "per:positive_impression": "positive impression",
    "per:negative_impression": "negative impression",
    "per:acquaintance": "acquaintance",
    "per:alumni": "alumni",
    "per:boss": "boss",
    "per:subordinate": "subordinate",
    "per:client": "client",
    "per:dates": "dates",
    "per:friends": "friends",
    "per:girl/boyfriend": "girlfriend or boyfriend",
    "per:neighbor": "neighbor",
    "per:roommate": "roommate",
    "per:children": "children",
    "per:other_family": "other family",
    "per:parents": "parents",
    "per:siblings": "siblings",
    "per:spouse": "spouse",
    "per:place_of_residence": "place of residence",
    "per:place_of_birth": "place of birth",
    "per:visited_place": "visited place",
    "per:origin": "origin",
    "per:employee_or_member_of": "employee or member of",
    "per:schools_attended": "schools attended",
    "per:works": "works",
    "per:age": "age",
    "per:date_of_birth": "date of birth",
    "per:major": "major",
    "per:place_of_work": "place of work",
    "per:title": "title",
    "per:alternate_names": "alternate names",
    "per:pet": "pet",
    "gpe:residents_of_place": "residents of place",
    "gpe:visitors_of_place": "visitors of place",
    "gpe:births_in_place": "births in place",
    "org:employees_or_members": "employees or members",
    "org:students": "students",
    "unanswerable": "unanswerable",
}

class Friends_dataset(TLiDB_Dataset):
    """
    Friends dataset
    # TODO: Fill in metadata
    
    """

    _dataset_name = "Friends"
    _tasks = [
        'emory_emotion_recognition', 'reading_comprehension', 'character_identification',
        'question_answering', 'personality_detection', 'relation_extraction',
        'MELD_emotion_recognition', 'response_generation'
    ]
    _url = "https://drive.google.com/uc?export=download&id=1QK_XX-d38fKeJlcTcoMT9ku3VZn6bv6H"
    _task_metadatas = {
        "emory_emotion_recognition": {
                "prompt":"emotion:","type":"classification","loader":"utterance_level_classification",
                "collate_type":"classification", "max_decode_tokens":5
        },
        "reading_comprehension": {
            "prompt": "Out of {}, [PLACEHOLDER] is ","type":"span_extraction",
            "loader": "character_span_extraction", "collate_type":"character_span_extraction",
            "max_decode_tokens":20
        },
        "character_identification": {
            "prompt": "Out of {}, '{}' in the phrase '{}' refers to", "type":"classification",
            "loader": "character_identification", "collate_type":"character_identification",
            "max_decode_tokens":5
        },
        "question_answering": {
            "prompt":"","type":"span_extraction","loader":"span_extraction",
            "collate_type":"span_extraction", "max_decode_tokens":64
        },
        "personality_detection": {
            "prompt":"{} is {}:",# <character> is <characteristic>
            "type":"multioutput_classification","num_outputs":5,"num_labels":2,
            "loader":"personality_detection","collate_type":"personality_detection",
            "max_decode_tokens":1, "label_map": {"0":"false", "1":"true"},
            "classes":["aggreeable", "conscientious", "extroverted", "open", "neurotic"]
        },
        "relation_extraction": {
            "prompt":"{} has the following relations with {}:","type":"multilabel_classification",
            "loader":"relation_extraction","collate_type":"relation_extraction",
            "max_decode_tokens":15, "class_to_natural_language_map": relations_to_natural_language_map,
            "default_prediction":"per:positive_impression"
        },
        "MELD_emotion_recognition": {
            "prompt":"emotion:", "type":"classification", "loader":"utterance_level_classification",
            "collate_type":"classification", "max_decode_tokens":5
        },
        "response_generation": {
            "prompt": "", "type":"response_generation","loader":"response_generation",
            "max_decode_tokens":128
        }
    }
    def __init__(self, task, dataset_folder, model_type, split, max_dialogue_length=None, few_shot_percent=None):
        assert task in self._tasks, f"{task} is not a valid task for {self._dataset_name}"
        super().__init__(self._dataset_name, task, model_type, max_dialogue_length, dataset_folder=dataset_folder)
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
                        truncated_dialogue = self._truncate_dialogue(dialogue)
                        str_dialogue = self._convert_dialogue_to_string(truncated_dialogue)
                        if task in turn:
                            self._input_array.append(str_dialogue)
                            self._y_array.append(turn[task])

    def _load_character_span_extraction_task(self, task, split_ids):
        for datum in self.dataset['data']:
            if datum['dialogue_id'] in split_ids:
                if task in datum['dialogue_metadata']:
                    dialogue = [[" ".join(turn['speakers']), turn['utterance']] for turn in datum['dialogue']]
                    truncated_dialogue = self._truncate_dialogue(dialogue)
                    str_dialogue = self._convert_dialogue_to_string(truncated_dialogue)

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

        def _create_truncated_dialogue_for_answer_turn(dialogue, answer_turn):
            """Truncate the dialogue by removing from the far end of the answer span"""
            start_turn_idx = 0
            if self.max_dialogue_length:
                while len(self._convert_dialogue_to_string(dialogue).split()) > self.max_dialogue_length:
                    # if answer is in first half of truncated dialogue, remove a turn from the end
                    if answer_turn-start_turn_idx < len(dialogue)/2:
                        dialogue = dialogue[:-1]
                    # if answer is in second half of truncated dialogue, remove a turn from the beginning
                    else:
                        dialogue = dialogue[1:]
                        start_turn_idx += 1
            return dialogue, start_turn_idx

        for datum in self.dataset['data']:
            if datum['dialogue_id'] in split_ids:
                if task in datum['dialogue_metadata']:
                    dialogue = [[" ".join(turn['speakers']), turn['utterance']] for turn in datum['dialogue']]
                    for qas in datum[task]:
                        for qa in qas['qas']:
                            answers = []
                            for answer in qa['answers']:
                                truncated_dialogue, start_turn_idx = _create_truncated_dialogue_for_answer_turn(dialogue.copy(), answer['answer_utterance_id'])
                                str_dialogue = self._convert_dialogue_to_string(truncated_dialogue)
                                answer_utterance_id = answer['answer_utterance_id']-start_turn_idx
                                
                                prior_turns = self._convert_dialogue_to_string(truncated_dialogue[:answer_utterance_id])
                                
                                # add 1 for space between prior turns and current turn
                                if prior_turns:
                                    answer_start = len(prior_turns) + 1
                                else:
                                    answer_start = 0

                                if not answer['is_speaker']:
                                    if truncated_dialogue[answer_utterance_id][0]:
                                        # if there is a speaker, add speaker + 2 for ': ' between current speaker and current turn
                                        answer_start += len(truncated_dialogue[answer_utterance_id][0]) + 2
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
                        truncated_dialogue = self._truncate_dialogue(dialogue)
                        str_dialogue = self._convert_dialogue_to_string(truncated_dialogue)

                        # encoder models can make use of multi-class classification significantly better
                        #   than decoder models, so encoder models get a single sample for all 5 classes
                        if self.model_type == "Encoder":
                            self._input_array.append({
                                "context": str_dialogue,
                                "focus_speaker": sample['focus_speaker']
                            })
                            self._y_array.append(sample['personality_characteristics'])

                        # decoder models get 5 separate samples, 1 for each class
                        else:
                            for characteristic, value in sample['personality_characteristics'].items():
                                self._input_array.append({
                                    "context": str_dialogue,
                                    "focus_speaker": sample['focus_speaker'],
                                    "class": characteristic
                                })
                                self._y_array.append(value)



    def _load_relation_extraction_task(self, task, split_ids):
        for datum in self.dataset['data']:
            if datum['dialogue_id'] in split_ids:
                if task in datum['dialogue_metadata']:
                    for sample in datum[task]:
                        dialogue = [[" ".join(datum['dialogue'][i]['speakers']), datum['dialogue'][i]['utterance']] for i in sample['turns']]
                        truncated_dialogue = self._truncate_dialogue(dialogue)
                        str_dialogue = self._convert_dialogue_to_string(truncated_dialogue)
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
        
    def _collate_decoder(self, batch):
        X, y, metadata = [], [], {}
        for item in batch:
            if self._task_metadata['collate_type'] == "classification":
                dialogue = item[0]
                X.append(self._join_strings("context:",dialogue,self._task_metadata['prompt']))
            elif self._task_metadata['collate_type'] == "character_span_extraction":
                X.append(self._join_strings("context:",item[0]['context'],"question:",item[0]['question'],self._task_metadata['prompt'].format(item[0]['entity_list'])))
            elif self._task_metadata['collate_type'] == "character_identification":
                truncated_context = self._truncate_dialogue(item[0]['context'])
                context = self._convert_dialogue_to_string(truncated_context)
                prompt = self._task_metadata['prompt'].format(" or ".join(self.task_labels), item[0]['entity_reference'],item[0]['entity_context'])
                X.append(self._join_strings("context:",context,prompt))
            elif self._task_metadata['collate_type'] == "span_extraction":
                X.append(self._join_strings("context:",item[0]['context'],
                                            "question:", item[0]['question'], "answer:"))
            elif self._task_metadata['collate_type'] == "personality_detection":
                prompt = self._task_metadata['prompt'].format(item[0]['focus_speaker'], item[0]['class'])
                X.append(self._join_strings("context:",item[0]['context'],prompt))
            elif self._task_metadata['collate_type'] == "relation_extraction":
                prompt = self._task_metadata['prompt'].format(item[0]['head'], item[0]['tail'])
                X.append(self._join_strings("context:",item[0]['context'],prompt))
            else:
                raise NotImplementedError(f"Collate type {self._task_metadata['collate_type']} not implemented")
            y.append(item[1])
            for k,v in item[2].items():
                if k not in metadata:
                    metadata.append(k)
                metadata[k].append(v)
        labels = self.task_labels
        if labels:
            metadata['labels'] = labels
        return X, y, metadata

    def _collate_encoderdecoder(self, batch):
        X, y, metadata = [], [], {}
        for item in batch:
            if self._task_metadata['collate_type'] == "classification":
                dialogue = item[0]
                X.append(self._join_strings("context:",dialogue,self._task_metadata['prompt']))
            elif self._task_metadata['collate_type'] == "character_span_extraction":
                X.append(self._join_strings("context:",item[0]['context'],"question:",item[0]['question'],self._task_metadata['prompt'].format(item[0]['entity_list'])))
            elif self._task_metadata['collate_type'] == "character_identification":
                truncated_context = self._truncate_dialogue(item[0]['context'])
                context = self._convert_dialogue_to_string(truncated_context)
                prompt = self._task_metadata['prompt'].format(" or ".join(self.task_labels), item[0]['entity_reference'],item[0]['entity_context'])
                X.append(self._join_strings("context:",context,prompt))
            elif self._task_metadata['collate_type'] == "span_extraction":
                X.append(self._join_strings("context:",item[0]['context'],
                                            "question:", item[0]['question'], "answer:"))
            elif self._task_metadata['collate_type'] == "personality_detection":
                prompt = self._task_metadata['prompt'].format(item[0]['focus_speaker'], item[0]['class'])
                X.append(self._join_strings("context:",item[0]['context'],prompt))
            elif self._task_metadata['collate_type'] == "relation_extraction":
                prompt = self._task_metadata['prompt'].format(item[0]['head'], item[0]['tail'])
                X.append(self._join_strings("context:",item[0]['context'],prompt))
            else:
                raise NotImplementedError(f"Collate type {self._task_metadata['collate_type']} not implemented")
            y.append(item[1])
            for k,v in item[2].items():
                if k not in metadata:
                    metadata.append(k)
                metadata[k].append(v)
        labels = self.task_labels
        if labels:
            metadata['labels'] = labels
        return X, y, metadata