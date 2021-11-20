#!/bin/bash

readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

readonly DD_URL='http://yanran.li/files/ijcnlp_dailydialog.zip'
readonly DD_FILE="${THIS_DIR}/dailydialog-raw.zip"
readonly DD_DIR="${THIS_DIR}/ijcnlp_dailydialog"

readonly RECCON_TRAIN_URL='https://raw.githubusercontent.com/declare-lab/RECCON/main/data/original_annotation/dailydialog_train.json'
readonly RECCON_VAL_URL='https://raw.githubusercontent.com/declare-lab/RECCON/main/data/original_annotation/dailydialog_valid.json'
readonly RECCON_TEST_URL='https://raw.githubusercontent.com/declare-lab/RECCON/main/data/original_annotation/dailydialog_test.json'

readonly RECCON_TRAIN_SP_EX_URL='https://raw.githubusercontent.com/declare-lab/RECCON/main/data/subtask1/fold1/dailydialog_qa_train_with_context.json'
readonly RECCON_VAL_SP_EX_URL='https://raw.githubusercontent.com/declare-lab/RECCON/main/data/subtask1/fold1/dailydialog_qa_valid_with_context.json'
readonly RECCON_TEST_SP_EX_URL='https://raw.githubusercontent.com/declare-lab/RECCON/main/data/subtask1/fold1/dailydialog_qa_test_with_context.json'

readonly RECCON_TRAIN_ENT_URL='https://raw.githubusercontent.com/declare-lab/RECCON/main/data/subtask2/fold1/dailydialog_classification_train_with_context.csv'
readonly RECCON_VAL_ENT_URL='https://raw.githubusercontent.com/declare-lab/RECCON/main/data/subtask2/fold1/dailydialog_classification_valid_with_context.csv'
readonly RECCON_TEST_ENT_URL='https://raw.githubusercontent.com/declare-lab/RECCON/main/data/subtask2/fold1/dailydialog_classification_test_with_context.csv'

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


   echo "Adding topic annotations"
   python3 gather_topics.py
   echo "Converting Daily Dialog to TLiDB format"
   python3 convert_daily_dialogue.py
   echo "Adding improved dialogue/emotion annotations from RECCON"
   python3 add_reccon_improved_annotations.py
   echo "Adding causal emotion annotations from RECCON"
   python3 add_reccon_causal_emotion_annotations.py

   zip -r TLiDB_Daily_Dialogue.zip TLiDB_Daily_Dialogue/
   rm -r "${DD_DIR}"
   rm -r "test"
   rm -r "train"
   rm -r "validation"
   for file in $DD_FILE "RECCON_train.json" "RECCON_validation.json" "RECCON_test.json" "RECCON_span_extraction_train.json" "RECCON_span_extraction_validation.json" "RECCON_span_extraction_test.json" "RECCON_entailment_train.csv" "RECCON_entailment_validation.csv" "RECCON_entailment_test.csv"
   do
      rm $file
   done


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