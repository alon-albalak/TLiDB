#!/bin/bash

readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

readonly DD_URL='http://yanran.li/files/ijcnlp_dailydialog.zip'
readonly DD_FILE="${THIS_DIR}/dailydialog-raw.zip"
readonly DD_DIR="${THIS_DIR}/ijcnlp_dailydialog"

readonly RECCON_TRAIN_URL='https://raw.githubusercontent.com/declare-lab/RECCON/main/data/original_annotation/dailydialog_train.json'
readonly RECCON_VAL_URL='https://raw.githubusercontent.com/declare-lab/RECCON/main/data/original_annotation/dailydialog_valid.json'
readonly RECCON_TEST_URL='https://raw.githubusercontent.com/declare-lab/RECCON/main/data/original_annotation/dailydialog_test.json'

function main() {
   trap exit SIGINT

   check_requirements

   fetch_file "${DD_URL}" "${DD_FILE}" 'Daily Dialog Data'
   extract_zip "${DD_FILE}" "${DD_DIR}" 'Daily Dialog Data'

   fetch_file "${RECCON_TRAIN_URL}" "RECCON_train.json"
   fetch_file "${RECCON_VAL_URL}" "RECCON_validation.json"
   fetch_file "${RECCON_TEST_URL}" "RECCON_test.json"

   python3 gather_topics.py
   python3 convert_daily_dialogue.py

   # zip -r TLiDB_Daily_Dialogue.zip TLiDB_Daily_Dialogue/
   # rm -r "${DD_DIR}"
   # rm -r "test"
   # rm -r "train"
   # rm -r "validation"
   # rm "${DD_FILE}"


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