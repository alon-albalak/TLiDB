# FETA Benchmark

FETA is a benchmark for few-sample task transfer in open-domain dialogue. FETA contains two underlying sets of conversations upon which there are 10 and 7 tasks annotated, enabling the study of intra-dataset task transfer; task transfer without domain adaptation. FETA's two datasets cover a variety of properties (dyadic vs. multi-party, anonymized vs. recurring speaker, varying dialogue lengths) and task types (utterance-level classification, dialogue-level classification, span extraction, multiple choice), and maintain a wide variety of data quantities.

## FETA Competition
The FETA competition is being hosted at the [5th Workshop on NLP For Conversational AI](https://sites.google.com/view/5thnlp4convai) (co-located with ACL 2023) and aims to encourage the development of new approaches to task-transfer with limited in-domain data.
<br>
Each of the FETA datasets has it's own submission portal ([FETA-DailyDialog](https://codalab.lisn.upsaclay.fr/competitions/10745) and [FETA-Friends](https://codalab.lisn.upsaclay.fr/competitions/10744)).
<br>
Participants are not required to compete in both datasets.

## Important Dates
- *Competition Start*: February 15th 2023 AoE
- *Competition End*: July 1st 2023 AoE (End of submissions to codalab)
- *Paper Submissions Due*: July 8th 2023 AoE
- *Awards and Prizes Announced*: Jully 12/13 2023

## How FETA Measures Transfer
The FETA Benchmark aims to measure the ability of a given model and learning algorithm to transfer knowledge from a source task(s) to a target task. FETA measures transfer by calculating 2 scores: a baseline score that does not utilize transfer, and a score for a method that utilizes transfer.
<br>
To be concrete, if a BERT model directly fine-tuned on emotion recognition gets 50% accuracy (baseline score), and a BERT model which first trains on question answering and then fine-tunes on emotion recognition gets 52% accuracy (transfer score), then FETA calculates a score delta of 2%.

## Awards and Prizes
**The FETA Benchmark Challenge will feature prizes for multiple teams**
<br>
First, the team with best overall transfer score for both datasets will receive a prize. Smaller awards will be given to winners of individual datasets.
<br>
Additionally, prizes will be given for innovative approaches (judged by an independent panel, based on submitted papers).
<br>
Exact awards are to be determined. Awards will be either cash prizes or GPUs given out at the workshop.


## Rules
### What data can I use?
In the FETA challenge, you are allowed to use any data or model that you would like, prior to computing your baseline score.
FETA Focuses on task transfer in a constrained setting and does not permit the use of outside data post-baseline.

### How do I submit my paper describing my method?
To be considered for prizes at the 5th Workshop on NLP for Conversational AI (co-located with ACL 2023) you must submit a paper describing the methods used in your final submission. The paper should be between 2-6 pages long. To submit, simply upload your work to ArXiv and send the link to feta.benchmark@gmail.com. Your paper should include all information necessary to understand the methods used, including:
- the baseline model
- any data used prior to training on FETA
- the training procedure
- the prompts/instructions/demonstrations used (when applicable)
- any preprocessing steps
- any hyperparameters
- any other information relevant to your submission



## Sample Code
See [tlidb/examples/FETA_sample_experiments.sh](tlidb/examples/FETA_sample_experiments.sh) for the code used to produce the results for a multi-task baseline that utilizes the a single source task for each target. This code takes about 4 hours to run on an NVIDIA RTX A6000 GPU.
<br>
Following training, see [tlidb/examples/FETA_create_submission.py](tlidb/examples/FETA_create_submission.py) for a simple script that organizes and zips prediction files into the expected submission format.
<br>
An example submission folder is found at [tlidb/examples/example_friends_submission](tlidb/examples/example_friends_submission). 


## Submission Format
Each submission should be composed of a folder for each target dataset. Within the folder should be 2 csv files containing model predictions. The first file is the baseline prediction file (named baseline_predictions.csv) and the second file is the final model predictions (predictions.csv).
For prediction format examples see the files in [tlidb/examples/example_friends_submission](tlidb/examples/example_friends_submission).
<br>
Each dataset in FETA is a different competition and so the submission folders should be separate.
<br>

*For FETA-DailyDialog*, a complete submission must have 10 folders named after each of the tasks in FETA-DailyDialog (e.g. "emotion_recognition"). Each task folder must contain 2 prediction files: baseline_predictions.csv and predictions.csv. The baseline_predictions.csv file contains the predictions from the baseline model, and the predictions.csv file contains the predictions from your final model. Each prediction file should have 2 columns, where the first column contains instance ids, and the second column contains the models predictions. Each prediction file should be in .csv format, and should NOT have a header row. Each task is different and has different expected prediction formats.
- adversarial_response_selection predictions are integers between 0 and 2
- causal_emotion_entailment predictions are integers between 0 and 1
- causal_emotion_span_extraction predictions are strings
- dialogue_act_classification predictions are integers between 0 and 3
- dialogue_nli predictions are integers between 0 and 1
- dialogue_reasoning_commonsense_relation_prediction predictions are integers between 0 and 31
- dialogue_reasoning_multiple_choice_span_selection predictions are integers between 0 and 3
- dialogue_reasoning_span_extraction predictions are strings
- emotion_recognition predictions are integers between 0 and 6
- topic_classification predictions are integers between 0 and 9
<br>

*For FETA-Friends*, a complete submission must have 7 folders named after each of the tasks in FETA-Friends (e.g. "character_identification"). Each task folder must contain 2 prediction files: baseline_predictions.csv and predictions.csv. The baseline_predictions.csv file contains the predictions from the baseline model, and the predictions.csv file contains the predictions from your final model. Each prediction file should have 2 columns, where the first column contains instance ids, and the second column contains the models predictions. Each prediction file should be in .csv format, and should NOT have a header row. Each task is different and has different expected prediction formats.
- character_identification predictions are integers between 0 and 6
- emory_emotion_recognition predictions are integers between 0 and 6
- MELD_emotion_recognition predictions are integers between 0 and 6
- personality_detection predictions are integers, either 0 or 1
- question_answering predictions are strings
- reading_comprehension predictions are strings
- relation_extraction predictions are lists of 37 integers, either 0 or 1

**Note**: to create a valid submission zip all task folders together into the same zip file with zip -r submission.zip * starting within the directory containing the folders.

## Suggested Approaches
To get teams started, we provide some suggested ideas that stem from differing approaches and motivations.
<br>
- Improving model generalization:
  - Multitask learning
  - Instruction fine-tuning
  - Prompting
  - Source-task selection
- Algorithmic advances for:
  - Transfer learning
  - Meta-learning
  - Target task-aware training
  - Continued pre-training


## Questions or Comments?
Please email the FETA team at [feta.benchmark@gmail.com](mailto:feta.benchmark@gmail.com)
