from .TLiDB_dataset import TLiDB_Dataset, load_split_ids
import random

class DailyDialog_dataset(TLiDB_Dataset):
    """
    DailyDialog dataset
    This dataset contains all available annotations (currently) from:
        - DailyDialog - http://yanran.li/files/ijcnlp_dailydialog.zip
            - emotion recognition
            - dialog act classification
            - topic classification
        - RECCON - https://github.com/declare-lab/RECCON
            - causal emotion span extraction
            - causal entailment of emotion
        - CIDER - https://github.com/declare-lab/CIDER
            - dialogue level NLI
            - Dialogue reasoning span extraction
            - Dialogue reasoning multiple choice span selection
            - Commonsense relation prediction
        - DailyDialog++ - https://iitmnlp.github.io/DailyDialog-plusplus/
            - Adversarial response selection

    Metadata:
        13118 Total dialogues
        Dialogues per task:
            Emotion recognition: 13118 Dialogues
            Dialog act classification: 13118 Dialogues
            Topic classification: 13118 Dialogues
            Causal emotion span extraction: 1106 Dialogues
            Causal entailment of emotion: 1106 Dialogues
            Dialogue level NLI: 245 Dialogues
            Dialogue reasoning span extraction: 227 Dialogues
            Dialogue reasoning multiple choice span selection: 226 Dialogues
            Commonsense relation prediction: 245 Dialogues
            Adversarial response selection: 6880 Dialogues
    """
    _dataset_name = 'DailyDialog'
    _tasks = [
        'emotion_recognition', 'dialogue_act_classification', 'topic_classification',
        'causal_emotion_span_extraction', 'causal_emotion_entailment',
        'dialogue_nli', 'dialogue_reasoning_span_extraction', 'dialogue_reasoning_multiple_choice_span_selection',
        'dialogue_reasoning_commonsense_relation_prediction', 'adversarial_response_selection',
        'response_generation'
        ]
    _url = "https://drive.google.com/uc?export=download&id=1U9dUi16RbAprUiSBmEKnEpwk45USfnml"
    _task_metadatas = {
        "emotion_recognition": {
                "prompt":"emotion:","type":"classification","loader":"utterance_level_classification",
                "collate_type":"classification"
                },
        "dialogue_act_classification": {
            "prompt":"dialogue act:","type":"classification","loader":"utterance_level_classification",
            "collate_type":"classification"
            },
        "topic_classification":{
            "prompt":"topic:","type":"classification","loader":"dialogue_level_classification",
            "collate_type":"classification"
            },
        "causal_emotion_span_extraction":{# data already contains the prompt
            "prompt":"","type":"span_extraction","loader":"span_extraction",
            "collate_type":"span_extraction"
            },
        "causal_emotion_entailment":{
            "prompt":"causal emotion entailment:","type":"classification","loader":"causal_emotion_entailment",
            "collate_type":"nli"
            },
        "dialogue_nli":{
            "prompt":"entailment:","type":"classification","loader":"dialogue_nli",
            "collate_type":"nli"
        },
        "dialogue_reasoning_span_extraction":{
            "prompt":"", "type":"span_extraction","loader":"span_extraction",
            "collate_type":"span_extraction"
        },
        "dialogue_reasoning_multiple_choice_span_selection":{
            "prompt":"The correct option is", "type":"multiple_choice","loader":"multiple_choice",
            "collate_type":"multiple_choice", "num_choices":4
        },
        "dialogue_reasoning_commonsense_relation_prediction":{
            "prompt":"", "type":"classification","loader":"relation_extraction",
            "collate_type":"relation_extraction"
        },
        "adversarial_response_selection":{
            "prompt":"The correct option is","type":"multiple_choice","loader":"adversarial_response_selection",
            "collate_type":"multiple_choice", "num_choices":3
        },
        "response_generation":{
            "prompt":"", "type":"response_generation","loader":"response_generation",
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
                        dialogue.append([turn['speakers'][0], turn['utterance']])
                        str_dialogue = self._convert_dialogue_to_string(dialogue)
                        if task in turn:
                            self._input_array.append(str_dialogue)
                            self._y_array.append(turn[task])

    def _load_dialogue_level_classification_task(self, task, split_ids):
        for datum in self.dataset['data']:
            if datum['dialogue_id'] in split_ids:
                if task in datum['dialogue_metadata']:            
                    dialogue = [[turn['speakers'][0], turn['utterance']] for turn in datum['dialogue']]
                    str_dialogue = self._convert_dialogue_to_string(dialogue)
                    self._input_array.append(str_dialogue)
                    self._y_array.append(datum[task])
        
    def _load_span_extraction_task(self, task, split_ids):
        for datum in self.dataset['data']:
            if datum['dialogue_id'] in split_ids:
                if task in datum['dialogue_metadata']:
                    for qas in datum[task]:
                        for qa in qas['qas']:
                            for answer in qa['answers']:
                                self._input_array.append({
                                    "context":qas['context'],"question":qa['question']
                                })
                                # move the answer start back by len(question) if not impossible
                                if answer['answer_start'] > 0:
                                    answer['answer_start'] += len(qa['question'])+1
                                self._y_array.append(answer)

    def _load_causal_emotion_entailment_task(self, task, split_ids):
        for datum in self.dataset['data']:
            if datum['dialogue_id'] in split_ids:
                if task in datum['dialogue_metadata']:
                    for sample in datum[task]:
                        self._input_array.append({
                            "premise":sample['history'],
                            "hypothesis": f"{sample['causal_utterance']} causes {sample['emotion']} in {sample['target_utterance']}",
                        })
                        self._y_array.append(sample['labels'])

    def _load_dialogue_nli_task(self, task, split_ids):
        for datum in self.dataset['data']:
            if datum['dialogue_id'] in split_ids:
                if task in datum['dialogue_metadata']:
                    for sample in datum[task]:
                        self._input_array.append({
                            "premise": self._convert_dialogue_to_string([[turn['speakers'][0], turn['utterance']] for turn in datum['dialogue']]),
                            "hypothesis": f"{sample['head']} {sample['relation']} {sample['tail']}"
                        })
                        self._y_array.append(sample['label'])

    def _load_multiple_choice_task(self, task, split_ids):
        for datum in self.dataset['data']:
            if datum['dialogue_id'] in split_ids:
                if task in datum['dialogue_metadata']:
                    context = datum[task]['context']
                    mcqs = datum[task]['mcqs']
                    for q in mcqs:
                        self._input_array.append({
                            "context": context,
                            "question": q['question'],
                            "options": q['options']
                        })
                        self._y_array.append(q['label'])

    def _load_relation_extraction_task(self, task, split_ids):
        for datum in self.dataset['data']:
            if datum['dialogue_id'] in split_ids:
                if task in datum['dialogue_metadata']:
                    dialogue = [[turn['speakers'][0], turn['utterance']] for turn in datum['dialogue']]
                    dialogue = self._convert_dialogue_to_string(dialogue)
                    for sample in datum[task]:
                        self._input_array.append({
                            "context": dialogue,
                            "head": sample['head'],
                            "tail": sample['tail']
                        })
                        self._y_array.append(sample['relation'])
                
    def _load_adversarial_response_selection_task(self, task, split_ids):
        for datum in self.dataset['data']:
            if datum['dialogue_id'] in split_ids:
                if task in datum['dialogue_metadata']:
                    for sample in datum[task]:
                        context = []
                        for turn in datum['dialogue']:
                            if turn['turn_id'] in sample['context_turns']:
                                context.append([turn['speakers'][0], turn['utterance']])
                        context = self._convert_dialogue_to_string(context)
                        for pos_resp, random_neg_resp, adv_neg_resp in zip(sample['positive_responses'], sample['random_negative_responses'], sample['adversarial_negative_responses']):
                            # shuffle the options
                            options = [pos_resp, random_neg_resp, adv_neg_resp]
                            random.shuffle(options)
                            self._input_array.append({
                                "context": context,
                                "options": options,
                                "question":"Which option is the best response?"
                            })
                            self._y_array.append(str(options.index(pos_resp)))

    def _collate_encoder(self, batch):
        X, y, metadata = [], [], {}
        for item in batch:
            if self._task_metadata['collate_type'] == 'span_extraction':
                X.append(self._join_strings(item[0]['question'],item[0]['context']))
            elif self._task_metadata['collate_type'] == 'nli':
                X.append(self._join_strings(item[0]['premise'], item[0]['hypothesis']))
            elif self._task_metadata['collate_type'] == 'classification':
                X.append(item[0])
            elif self._task_metadata['collate_type'] == 'multiple_choice':
                mcq_inputs = []
                context = item[0]['context']
                question = item[0]['question']
                options = item[0]['options']
                for option in options:
                    mcq_inputs.append(self._join_strings(context,"[SEP]", question, "[SEP]", option))
                X.append(mcq_inputs)
            elif self._task_metadata['collate_type'] == "relation_extraction":
                X.append(self._join_strings(item[0]['head'],"[SEP]",item[0]['tail'],"[SEP]",item[0]['context']))
            else:
                raise NotImplementedError(f"Collate type {self._task_metadata['collate_type']} not implemented")
            y.append(item[1])
            for k, v in item[2].items():
                if k not in metadata:
                    metadata.append(k)
                metadata[k].append(v)
        return X, y, metadata

    def _collate_decoder(self, batch):
        X, y, metadata = [], [], {}
        for item in batch:
            if self._task_metadata['collate_type'] == 'span_extraction':
                X.append(self._join_strings("context:",item[0]['context'],
                                            "question:", item[0]['question'], "answer:"))
            elif self._task_metadata['collate_type'] == 'nli':
                X.append(self._join_strings("context:",item[0]['premise'],
                                            self._task_metadata['prompt'],item[0]['hypothesis']))
            elif self._task_metadata['collate_type'] == 'classification':
                dialogue = item[0]
                X.append(self._join_strings("context:",dialogue,self._task_metadata['prompt']))
            elif self._task_metadata['collate_type'] == 'multiple_choice':
                options_str = " ".join(f"option {i}: {option}" for i, option in enumerate(item[0]['options']))
                X.append(self._join_strings("context:",item[0]['context'],
                                            "question:",item[0]['question'],\
                                            options_str,self._task_metadata['prompt']))
            elif self._task_metadata['collate_type'] == "relation_extraction":
                X.append(self._join_strings("context:",item[0]['context'],
                                            f"The relation between '{item[0]['head']}' and '{item[0]['tail']}' is"))
            else:
                raise NotImplementedError(f"Collate type {self._task_metadata['collate_type']} not implemented")       
            y.append(item[1])
            for k, v in item[2].items():
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
            if self._task_metadata['collate_type'] == 'span_extraction':
                X.append(self._join_strings("context:",item[0]['context'],
                                            "question:", item[0]['question'], "answer:"))
            elif self._task_metadata['collate_type'] == 'nli':
                X.append(self._join_strings("context:",item[0]['premise'],
                                            self._task_metadata['prompt'],item[0]['hypothesis']))
            elif self._task_metadata['collate_type'] == 'classification':
                dialogue = item[0]
                X.append(self._join_strings("context:",dialogue,self._task_metadata['prompt']))
            elif self._task_metadata['collate_type'] == 'multiple_choice':
                options_str = " ".join(f"option {i}: {option}" for i, option in enumerate(item[0]['options']))
                X.append(self._join_strings("context:",item[0]['context'],
                                            "question:",item[0]['question'],\
                                            options_str,self._task_metadata['prompt']))
            elif self._task_metadata['collate_type'] == "relation_extraction":
                X.append(self._join_strings("context:",item[0]['context'],
                                            f"The relation between '{item[0]['head']}' and '{item[0]['tail']}' is"))
            else:
                raise NotImplementedError(f"Collate type {self._task_metadata['collate_type']} not implemented")       
            y.append(item[1])
            for k, v in item[2].items():
                if k not in metadata:
                    metadata.append(k)
                metadata[k].append(v)
        labels = self.task_labels
        if labels:
            metadata['labels'] = labels
        return X, y, metadata