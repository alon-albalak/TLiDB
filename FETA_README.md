# FETA Benchmark

FETA is a benchmark for few-sample task transfer in open-domain dialogue. FETA contains two underlying sets of conversations upon which there are 10 and 7 tasks annotated, enabling the study of intra-dataset task transfer; task transfer without domain adaptation. FETA's two datasets cover a variety of properties (dyadic vs. multi-party, anonymized vs. recurring speaker, varying dialogue lengths) and task types (utterance-level classification, dialogue-level classification, span extraction, multiple choice), and maintain a wide variety of data quantities.

# Competition Overview
### FETA Competition
The FETA competition is being hosted at the [5th Workshop on NLP For Conversational AI](https://sites.google.com/view/5thnlp4convai) (co-located with ACL 2023) and aims to encourage the development of new approaches to task-transfer with limited data.
<br>
Participation in the FETA challenge has 2 components: model predictions submitted to codalab (each of the FETA datasets has it's own submission portal ([FETA-DailyDialog](https://codalab.lisn.upsaclay.fr/competitions/10745) and [FETA-Friends](https://codalab.lisn.upsaclay.fr/competitions/10744)), and a paper describing your methods.
<br>
*NOTE*: Participants are not required to compete in both datasets.
<br>

### Important Dates
- *Competition Start*: February 15th 2023 AoE
- *Competition End*: July 1st 2023 AoE (End of submissions to codalab)
- *Paper Submissions Due*: July 8th 2023 AoE
- *Awards and Prizes Announced*: July 14 2023

### Awards and Prizes
**The FETA Benchmark Challenge will feature prizes for multiple teams**
<br>
First, the team with best overall transfer score for both datasets will receive a prize. Smaller awards will be given to winners of individual datasets.
<br>
Additionally, prizes will be given for innovative approaches (judged by an independent panel, based on submitted papers).
<br>
Exact awards are to be determined. Awards will be either cash prizes or GPUs given out at the workshop.

# Competition Details

### How FETA Measures Transfer
The FETA Benchmark aims to measure the ability of a given model and learning algorithm to transfer knowledge from a source task(s) to a target task. FETA measures transfer by calculating 2 scores: a baseline score that does not utilize transfer, and a score for a method that utilizes transfer.
<br>
To be concrete, if a BERT model directly fine-tuned on emotion recognition gets 50% accuracy (baseline score), and a BERT model which first trains on question answering and then fine-tunes on emotion recognition gets 52% accuracy (transfer score), then FETA calculates a score delta of 2%.

### How does intra-dataset task transfer work with FETA datasets?
FETA measures model-specific transfer. This requires that each model to be measured has both a **baseline score** and a **transfer score**, after utilizing knowledge transfer. We divide training into multiple steps to better understand how intra-dataset task transfer works in FETA.
<br><br>
**1 - Pre-FETA Training:** The first stage of training should occur prior to calculating baseline scores. At this stage, you are allowed to use any outside sources of data. In our [example code](tlidb/examples/FETA_sample_experiments.sh) we use a pre-trained BERT model, so the only data we utilize prior to baseline is BERT's pretraining corpus.
<br><br>
**2 - Baseline Model Training/Evaluation:** The next stage requires that we measure baseline scores. For each task in the FETA dataset, the baseline score is calculated by taking the model from stage 1 and fine-tuning and evaluating directly on the task. This step can be seen in the `train_eval_target` function of the [example code](tlidb/examples/FETA_sample_experiments.sh), where for each target task we first fine-tune and then evaluate the model.
<!-- <br>
*NOTE:* If you use instructions, demonstrations, or other prompting methods, the same instructions, demonstrations and prompts should be used for baseline scores and transfer scores. -->
**3 - Knowledge Transfer and Evaluation:** The third stage is applying knowledge transfer to each target task within the FETA dataset. You can continue with the model from stage 2, or start from scratch with the model in stage 1 (our examples perform transfer on the model from stage 1, a pre-trained BERT). You can use a single model for all target tasks, but we highly recommend repeating this stage for each target task in the dataset. This step requires that you use the few-shot data for each target task.
<br>
Finally, we measure transfer scores. This is done by evaluating the model directly on the test set. Our [example code](tlidb/examples/FETA_sample_experiments.sh) performs knowledge transfer and evaluation in the `multitask_finetune_eval_target` function.
<br>
*NOTE:* To properly measure intra-dataset task transfer, no outside data is allowed in this stage. For example, if we are working in FETA-Friends and the target task is question answering, then we can only use: the few-shot data for question answering and the few-shot data available for other FETA-Friends tasks. Intra-dataset task transfer does not allow for the use of DailyDialog when working on a Friends target dataset, or vice versa, and does not allow for any outside data to be used at this stage.

### Scoring
Each submission will be given 4 scores:
- Baseline Score: the baseline model's scores averaged across all tasks
- Transfer Score: the final model's scores averaged across all tasks (includes knowledge transfer)
- Score Delta: the difference between transfer score and baseline score
- Submission Score: The score on which a submission will be judged. It is calculated as the score delta plus one-tenth the baseline score (SD + 0.1 \* BS)

### How do I officially participate in the FETA Benchmark Challenge?
Participation in the FETA challenge has 2 components: your submissions to the [FETA-DailyDialog](https://codalab.lisn.upsaclay.fr/competitions/10745) and/or [FETA-Friends](https://codalab.lisn.upsaclay.fr/competitions/10744) codalab site (deadline: July 1st 2023 AoE), and a paper describing your methods emailed to us at [feta.benchmark@gmail.com](mailto:feta.benchmark@gmail.com) (deadline: July 8th 2023 AoE). The paper should use \*ACL format, available on [GitHub](https://github.com/acl-org/acl-style-files) and as an [Overleaf template](https://www.overleaf.com/project/5f64f1fb97c4c50001b60549). In the email submission of your paper, include your team name from codalab.


### How do I submit the paper describing my method?
To be considered for prizes at the 5th Workshop on NLP for Conversational AI (co-located with ACL 2023) you must submit a paper describing the methods used in your final submission. The paper should be between 2-6 pages long and use \*ACL format, available on [GitHub](https://github.com/acl-org/acl-style-files) or as an [Overleaf template](https://www.overleaf.com/project/5f64f1fb97c4c50001b60549). To submit, simply upload your work to ArXiv and send the link to feta.benchmark@gmail.com along with your team name. Your paper should include all information necessary to understand the methods used, including:
- the baseline model
- any data used prior to training on FETA
- the training procedure
- the prompts/instructions/demonstrations used (when applicable)
- any preprocessing steps
- any hyperparameters
- any other information relevant to your submission

### Sample Code
See [tlidb/examples/FETA_sample_experiments.sh](tlidb/examples/FETA_sample_experiments.sh) for the code used to produce the results for a multi-task baseline that utilizes the a single source task for each target. This code takes about 4 hours to run on an NVIDIA RTX A6000 GPU.
<br>
Following training, see [tlidb/examples/FETA_create_submission.py](tlidb/examples/FETA_create_submission.py) for a simple script that organizes and zips prediction files into the expected submission format.
<br>
An example submission folder is found at [tlidb/examples/example_friends_submission](tlidb/examples/example_friends_submission). 


### Codalab Submission Format
For each FETA dataset, a submission is composed of a folder for each target task. Each task folder must contain 2 prediction files: baseline_predictions.csv and predictions.csv. The baseline_predictions.csv file contains the predictions from the baseline model, and the predictions.csv file contains the predictions from the model utilizing knowlege transfer. Each prediction file should have 2 columns, where the first column contains instance ids, and the second column contains the models predictions. Each prediction file should be in .csv format, and should NOT have a header row. Each task is different and has different expected prediction formats.

For prediction format examples see the files in [tlidb/examples/example_friends_submission](tlidb/examples/example_friends_submission).
<br>
Each dataset in FETA is a different competition so submission folders should be separate.
<br>

*For FETA-DailyDialog*, a complete submission must have 10 folders named after each of the tasks in FETA-DailyDialog (e.g. "emotion_recognition"). The prediction formats for instances in each of the datasets is as follows: 
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

*For FETA-Friends*, a complete submission must have 7 folders named after each of the tasks in FETA-Friends (e.g. "character_identification"). The prediction formats for instances in each of the datasets is as follows:
- character_identification predictions are integers between 0 and 6
- emory_emotion_recognition predictions are integers between 0 and 6
- MELD_emotion_recognition predictions are integers between 0 and 6
- personality_detection predictions are integers, either 0 or 1
- question_answering predictions are strings
- reading_comprehension predictions are strings
- relation_extraction predictions are lists of 37 integers, either 0 or 1

**Note**: to create a valid submission zip all task folders together into the same zip file with zip -r submission.zip * starting within the directory containing the folders. See [tlidb/examples/example_friends_submission.zip](tlidb/examples/example_friends_submission.zip) for an example.

### Suggested Approaches
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

## Rules
<ol>
  <li>There are no restrictions on model sizes (# parameters) or model architectures.</li>
  <li>All data used (for both source and target tasks) must be from the few-shot dataset. If using the TLiDB training scripts, simply use the `--few_shot_percent=0.1` flag. If loading the data from your own script, use the files named `0.1_percent_few_shot_{train|dev}_ids`. **Do not use the full shot datasets**</li>
  <li>For an individual target task, you may use as many source tasks as you'd like as long as they are within the same dataset. For example, if we are working in FETA-Friends and the target task is question answering, then we can use anywhere between 1-6 of the remaining tasks from FETA-Friends as the source tasks.</li>
</ol>

### Questions or Comments?
Please email the FETA team at [feta.benchmark@gmail.com](mailto:feta.benchmark@gmail.com)
