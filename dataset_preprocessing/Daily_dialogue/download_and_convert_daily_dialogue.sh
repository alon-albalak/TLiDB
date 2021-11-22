#!/bin/bash

readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# ORIGINAL DAILY DIALOG FILES
readonly DD_URL='http://yanran.li/files/ijcnlp_dailydialog.zip'
readonly DD_FILE="${THIS_DIR}/dailydialog-raw.zip"
readonly DD_DIR="${THIS_DIR}/ijcnlp_dailydialog"

# RECCON FILES
readonly RECCON_TRAIN_URL='https://raw.githubusercontent.com/declare-lab/RECCON/main/data/original_annotation/dailydialog_train.json'
readonly RECCON_VAL_URL='https://raw.githubusercontent.com/declare-lab/RECCON/main/data/original_annotation/dailydialog_valid.json'
readonly RECCON_TEST_URL='https://raw.githubusercontent.com/declare-lab/RECCON/main/data/original_annotation/dailydialog_test.json'

readonly RECCON_TRAIN_SP_EX_URL='https://raw.githubusercontent.com/declare-lab/RECCON/main/data/subtask1/fold1/dailydialog_qa_train_with_context.json'
readonly RECCON_VAL_SP_EX_URL='https://raw.githubusercontent.com/declare-lab/RECCON/main/data/subtask1/fold1/dailydialog_qa_valid_with_context.json'
readonly RECCON_TEST_SP_EX_URL='https://raw.githubusercontent.com/declare-lab/RECCON/main/data/subtask1/fold1/dailydialog_qa_test_with_context.json'

readonly RECCON_TRAIN_ENT_URL='https://raw.githubusercontent.com/declare-lab/RECCON/main/data/subtask2/fold1/dailydialog_classification_train_with_context.csv'
readonly RECCON_VAL_ENT_URL='https://raw.githubusercontent.com/declare-lab/RECCON/main/data/subtask2/fold1/dailydialog_classification_valid_with_context.csv'
readonly RECCON_TEST_ENT_URL='https://raw.githubusercontent.com/declare-lab/RECCON/main/data/subtask2/fold1/dailydialog_classification_test_with_context.csv'

# CIDER FILES
readonly CIDER_MAIN_URL='https://raw.githubusercontent.com/declare-lab/CIDER/main/data/cider_main.json'
readonly CIDER_DNLI_TRAIN_URL='https://raw.githubusercontent.com/declare-lab/CIDER/main/code/dialogue_nli/data/fold1_w_neg_train_lemma.tsv'
readonly CIDER_DNLI_TEST_URL='https://raw.githubusercontent.com/declare-lab/CIDER/main/code/dialogue_nli/data/fold1_w_neg_test_lemma.tsv'

readonly CIDER_SP_EX_TRAIN_URL='https://raw.githubusercontent.com/declare-lab/CIDER/main/code/span_extraction/data/fold1_train.json'
readonly CIDER_SP_EX_TEST_URL='https://raw.githubusercontent.com/declare-lab/CIDER/main/code/span_extraction/data/fold1_test.json'

readonly CIDER_MCQ_T0_URL='https://raw.githubusercontent.com/declare-lab/CIDER/main/code/mcq/dataset/train_iter0.csv'
readonly CIDER_MCQ_T35_URL='https://raw.githubusercontent.com/declare-lab/CIDER/main/code/mcq/dataset/train_iter35.csv'
readonly CIDER_MCQ_V0_URL='https://raw.githubusercontent.com/declare-lab/CIDER/main/code/mcq/dataset/val_iter0.csv'
readonly CIDER_MCQ_V35_URL='https://raw.githubusercontent.com/declare-lab/CIDER/main/code/mcq/dataset/val_iter35.csv'

readonly CIDER_RP_URL='https://raw.githubusercontent.com/declare-lab/CIDER/main/code/relation_prediction/datasets/csk_w_neg_rp/relations.txt'
readonly CIDER_RP_TRAIN_URL='https://raw.githubusercontent.com/declare-lab/CIDER/main/code/relation_prediction/datasets/csk_w_neg_rp/fold1_w_neg_train_lemma.csv'
readonly CIDER_RP_TEST_URL='https://raw.githubusercontent.com/declare-lab/CIDER/main/code/relation_prediction/datasets/csk_w_neg_rp/fold1_w_neg_test_lemma.csv'

readonly DDpp_TRAIN_URL='https://iitmnlp.github.io/DailyDialog-plusplus/dataset/train.json'
readonly DDpp_DEV_URL='https://iitmnlp.github.io/DailyDialog-plusplus/dataset/dev.json'
readonly DDpp_TEST_URL='https://iitmnlp.github.io/DailyDialog-plusplus/dataset/test.json'

function main() {
   trap exit SIGINT

   check_requirements

   fetch_file "${DD_URL}" "${DD_FILE}" 'Daily Dialog Data'
   extract_zip "${DD_FILE}" "${DD_DIR}" 'Daily Dialog Data'

   # these files are not currently used, but contain the original annotated data from RECCON
   fetch_file "${RECCON_TRAIN_URL}" "RECCON_train.json"
   fetch_file "${RECCON_VAL_URL}" "RECCON_validation.json"
   fetch_file "${RECCON_TEST_URL}" "RECCON_test.json"

   fetch_file "${RECCON_TRAIN_SP_EX_URL}" "RECCON_span_extraction_train.json"
   fetch_file "${RECCON_VAL_SP_EX_URL}" "RECCON_span_extraction_validation.json"
   fetch_file "${RECCON_TEST_SP_EX_URL}" "RECCON_span_extraction_test.json"

   fetch_file "${RECCON_TRAIN_ENT_URL}" "RECCON_entailment_train.csv"
   fetch_file "${RECCON_VAL_ENT_URL}" "RECCON_entailment_validation.csv"
   fetch_file "${RECCON_TEST_ENT_URL}" "RECCON_entailment_test.csv"

   fetch_file "${CIDER_MAIN_URL}" "CIDER_main.json"
   fetch_file "${CIDER_DNLI_TRAIN_URL}" "CIDER_DNLI_train.tsv"
   fetch_file "${CIDER_DNLI_TEST_URL}" "CIDER_DNLI_test.tsv"

   fetch_file "${CIDER_SP_EX_TRAIN_URL}" "CIDER_sp_ex_train.json"
   fetch_file "${CIDER_SP_EX_TEST_URL}" "CIDER_sp_ex_test.json"

   fetch_file "${CIDER_MCQ_T0_URL}" "CIDER_MCQ_train_iter0.csv"
   fetch_file "${CIDER_MCQ_T35_URL}" "CIDER_MCQ_train_iter35.csv"
   fetch_file "${CIDER_MCQ_V0_URL}" "CIDER_MCQ_val_iter0.csv"
   fetch_file "${CIDER_MCQ_V35_URL}" "CIDER_MCQ_val_iter35.csv"

   fetch_file "${CIDER_RP_URL}" "CIDER_RP_relations.txt"
   fetch_file "${CIDER_RP_TRAIN_URL}" "CIDER_RP_train.csv"
   fetch_file "${CIDER_RP_TEST_URL}" "CIDER_RP_test.csv"

   fetch_file "${DDpp_TRAIN_URL}" "DDpp_train.json"
   fetch_file "${DDpp_DEV_URL}" "DDpp_dev.json"
   fetch_file "${DDpp_TEST_URL}" "DDpp_test.json"


   echo "Adding topic annotations"
   python3 gather_topics.py
   echo "Converting Daily Dialog to TLiDB format"
   python3 convert_daily_dialogue.py
   echo "Adding improved dialogue/emotion annotations from RECCON and DailyDialog++"
   python3 add_improved_annotations.py
   echo "Adding causal emotion annotations from RECCON"
   python3 add_reccon_causal_emotion_annotations.py
   echo "Adding dialogue reasoning annotations from CIDER"
   python3 add_cider_annotations.py
   # echo "Adding adversarial response selection annotations from DailyDialog++"

   # zip -r TLiDB_Daily_Dialogue.zip TLiDB_Daily_Dialogue/
   # rm -r "${DD_DIR}"
   # rm -r "test"
   # rm -r "train"
   # rm -r "validation"
   # for file in $DD_FILE "RECCON_train.json" "RECCON_validation.json" "RECCON_test.json" "RECCON_span_extraction_train.json" "RECCON_span_extraction_validation.json" "RECCON_span_extraction_test.json" "RECCON_entailment_train.csv" "RECCON_entailment_validation.csv" "RECCON_entailment_test.csv"
   # do
   #    rm $file
   # done


}

function check_requirements() {
   local hasWget
   local hasCurl

   type wget > /dev/null 2> /dev/null
   hasWget=$?

   type curl > /dev/null 2> /dev/null
   hasCurl=$?

   if [[ "${hasWget}" -ne 0 ]] && [[ "${hasCurl}" -ne 0 ]]; then
      echo 'ERROR: wget or curl required to download dataset'
      exit 10
   fi

   type tar > /dev/null 2> /dev/null
   if [[ "$?" -ne 0 ]]; then
      echo 'ERROR: tar required to extract dataset'
      exit 11
   fi
}

function get_fetch_command() {
   type curl > /dev/null 2> /dev/null
   if [[ "$?" -eq 0 ]]; then
      echo "curl -L -o"
      return
   fi

   type wget > /dev/null 2> /dev/null
   if [[ "$?" -eq 0 ]]; then
      echo "wget -O"
      return
   fi

   echo 'ERROR: wget or curl not found'
   exit 20
}

function fetch_file() {
   local url=$1
   local path=$2
   local name=$3

   if [[ -e "${path}" ]]; then
      echo "${name} file found cached, skipping download."
      return
   fi

   echo "Downloading ${name} file with command: $FETCH_COMMAND"
   `get_fetch_command` "${path}" "${url}"
   if [[ "$?" -ne 0 ]]; then
      echo "ERROR: Failed to download ${name} file"
      exit 30
   fi
}

function extract_zip() {
   local path=$1
   local expectedDir=$2
   local name=$3

   if [[ -e "${expectedDir}" ]]; then
      echo "Extracted ${name} zip found cached, skipping extract."
      return
   fi

   echo "Extracting the ${name} zip"
   unzip -d "$(dirname ${expectedDir})" "${path}"
   unzip "${expectedDir}/test.zip"
   unzip "${expectedDir}/train.zip"
   unzip "${expectedDir}/validation.zip"

   if [[ "$?" -ne 0 ]]; then
      echo "ERROR: Failed to extract ${name} zip"
      exit 40
   fi
}

main "$@"