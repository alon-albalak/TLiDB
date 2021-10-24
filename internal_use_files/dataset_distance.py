def gather_utterances(dataset):
    utterances = []
    for sample in dataset['data']:
        for utterance in sample['dialogue']:
            utterances.append(utterance['utterance'])
    return utterances