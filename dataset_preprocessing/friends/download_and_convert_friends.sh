#!/bin/bash

readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# emoryNLP character mining scripts
readonly emoryNLP_1_URL='https://raw.githubusercontent.com/emorynlp/character-mining/master/json/friends_season_01.json'
readonly emoryNLP_2_URL='https://raw.githubusercontent.com/emorynlp/character-mining/master/json/friends_season_02.json'
readonly emoryNLP_3_URL='https://raw.githubusercontent.com/emorynlp/character-mining/master/json/friends_season_03.json'
readonly emoryNLP_4_URL='https://raw.githubusercontent.com/emorynlp/character-mining/master/json/friends_season_04.json'
readonly emoryNLP_5_URL='https://raw.githubusercontent.com/emorynlp/character-mining/master/json/friends_season_05.json'
readonly emoryNLP_6_URL='https://raw.githubusercontent.com/emorynlp/character-mining/master/json/friends_season_06.json'
readonly emoryNLP_7_URL='https://raw.githubusercontent.com/emorynlp/character-mining/master/json/friends_season_07.json'
readonly emoryNLP_8_URL='https://raw.githubusercontent.com/emorynlp/character-mining/master/json/friends_season_08.json'
readonly emoryNLP_9_URL='https://raw.githubusercontent.com/emorynlp/character-mining/master/json/friends_season_09.json'
readonly emoryNLP_10_URL='https://raw.githubusercontent.com/emorynlp/character-mining/master/json/friends_season_10.json'
readonly emoryNLP_DIR="${THIS_DIR}/emoryNLP"

# emoryNLP friends emotion recognition dataset
# readonly ER_TRAIN_URL='https://raw.githubusercontent.com/emorynlp/emotion-detection/master/json/emotion-detection-trn.json'
# readonly ER_DEV_URL='https://raw.githubusercontent.com/emorynlp/emotion-detection/master/json/emotion-detection-dev.json'
# readonly ER_TEST_URL='https://raw.githubusercontent.com/emorynlp/emotion-detection/master/json/emotion-detection-tst.json'

# # emoryNLP friends reading comprehension dataset
# readonly RC_TRAIN_URL='https://raw.githubusercontent.com/emorynlp/reading-comprehension/master/json/reading-comprehension-trn.json'
# readonly RC_DEV_URL='https://raw.githubusercontent.com/emorynlp/reading-comprehension/master/json/reading-comprehension-dev.json'
# readonly RC_TEST_URL='https://raw.githubusercontent.com/emorynlp/reading-comprehension/master/json/reading-comprehension-tst.json'

# emoryNLP friends question answering dataset
readonly QA_TRAIN_URL='https://raw.githubusercontent.com/emorynlp/FriendsQA/master/dat/friendsqa_trn.json'
readonly QA_DEV_URL='https://raw.githubusercontent.com/emorynlp/FriendsQA/master/dat/friendsqa_dev.json'
readonly QA_TEST_URL='https://raw.githubusercontent.com/emorynlp/FriendsQA/master/dat/friendsqa_tst.json'

# Emotion Recognition Dataset
readonly MELD_URL='http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz'
readonly MELD_FILE="${THIS_DIR}/MELD.Raw.tar.gz"
readonly MELD_DIR="${THIS_DIR}/MELD"


function main() {
    trap exit SIGINT

    check_requirements

   # download emoryNLP character mining datasets
   mkdir "${emoryNLP_DIR}"
   fetch_file "${emoryNLP_1_URL}" "${emoryNLP_DIR}/friends_season_01.json" "EmoryNLP Friends Season 1"
   fetch_file "${emoryNLP_2_URL}" "${emoryNLP_DIR}/friends_season_02.json" "EmoryNLP Friends Season 2"
   fetch_file "${emoryNLP_3_URL}" "${emoryNLP_DIR}/friends_season_03.json" "EmoryNLP Friends Season 3"
   fetch_file "${emoryNLP_4_URL}" "${emoryNLP_DIR}/friends_season_04.json" "EmoryNLP Friends Season 4"
   fetch_file "${emoryNLP_5_URL}" "${emoryNLP_DIR}/friends_season_05.json" "EmoryNLP Friends Season 5"
   fetch_file "${emoryNLP_6_URL}" "${emoryNLP_DIR}/friends_season_06.json" "EmoryNLP Friends Season 6"
   fetch_file "${emoryNLP_7_URL}" "${emoryNLP_DIR}/friends_season_07.json" "EmoryNLP Friends Season 7"
   fetch_file "${emoryNLP_8_URL}" "${emoryNLP_DIR}/friends_season_08.json" "EmoryNLP Friends Season 8"
   fetch_file "${emoryNLP_9_URL}" "${emoryNLP_DIR}/friends_season_09.json" "EmoryNLP Friends Season 9"
   fetch_file "${emoryNLP_10_URL}" "${emoryNLP_DIR}/friends_season_10.json" "EmoryNLP Friends Season 10"

   # convert original emoryNLP files into TLiDB format w/ 
   #      emotion recognition and reading comprehension datasets
   python3 convert_emoryNLP.py

    # Do not delete this file, it takes a long time to download
    # fetch_file "${MELD_URL}" "${MELD_FILE}" 'MELD Data'
    # extract_tar "${MELD_FILE}" "${MELD_DIR}" 'MELD Data'

   # download emoryNLP Question Answering dataset
   fetch_file "${QA_TRAIN_URL}" "emory_question_answering_train.json" 'EmoryNLP Friends Question Answering Dataset'
   fetch_file "${QA_DEV_URL}" "emory_question_answering_dev.json" 'EmoryNLP Friends Question Answering Dataset'
   fetch_file "${QA_TEST_URL}" "emory_question_answering_test.json" 'EmoryNLP Friends Question Answering Dataset'
   python3 add_friends_QA_annotations.py

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

function extract_tar() {
   local path=$1
   local expectedDir=$2
   local name=$3

   if [[ -e "${expectedDir}" ]]; then
      echo "Extracted ${name} zip found cached, skipping extract."
      return
   fi

   mkdir -p "${expectedDir}"

   echo "Extracting the ${name} zip"
   tar -xvf "${path}" -C "${expectedDir}"
   #unzip -d "$(dirname ${expectedDir})" "${path}"
   #unzip "${expectedDir}/test.zip"
   #unzip "${expectedDir}/train.zip"
   #unzip "${expectedDir}/validation.zip"

   if [[ "$?" -ne 0 ]]; then
      echo "ERROR: Failed to extract ${name} zip"
      exit 40
   fi
}

main "$@"