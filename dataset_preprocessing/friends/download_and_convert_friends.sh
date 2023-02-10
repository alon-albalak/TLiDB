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

# emoryNLP friends question answering dataset
readonly QA_TRAIN_URL='https://raw.githubusercontent.com/emorynlp/FriendsQA/master/dat/friendsqa_trn.json'
readonly QA_DEV_URL='https://raw.githubusercontent.com/emorynlp/FriendsQA/master/dat/friendsqa_dev.json'
readonly QA_TEST_URL='https://raw.githubusercontent.com/emorynlp/FriendsQA/master/dat/friendsqa_tst.json'

# emoryNLP friends personality detection dataset
readonly PD_URL='https://raw.githubusercontent.com/emorynlp/personality-detection/master/CSV/friends-personality.csv'

# Emotion Recognition Dataset
readonly MELD_TRAIN_URL='https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv'
readonly MELD_DEV_URL='https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/dev_sent_emo.csv'
readonly MELD_TEST_URL='https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/test_sent_emo.csv'

# DialogRE Dataset
readonly DIALOGRE_URL='https://drive.google.com/uc?export=download&id=1D8TfIM1MfYG32m-sNy3tH4JLy6GQLo4j'


function main() {
    trap exit SIGINT

    check_requirements

   # download emoryNLP character mining datasets
   mkdir "${emoryNLP_DIR}"
   fetch_file "${emoryNLP_1_URL}" "${emoryNLP_DIR}/friends_season_01.json" "EmoryNLP Friends - Season 1"
   fetch_file "${emoryNLP_2_URL}" "${emoryNLP_DIR}/friends_season_02.json" "EmoryNLP Friends - Season 2"
   fetch_file "${emoryNLP_3_URL}" "${emoryNLP_DIR}/friends_season_03.json" "EmoryNLP Friends - Season 3"
   fetch_file "${emoryNLP_4_URL}" "${emoryNLP_DIR}/friends_season_04.json" "EmoryNLP Friends - Season 4"
   fetch_file "${emoryNLP_5_URL}" "${emoryNLP_DIR}/friends_season_05.json" "EmoryNLP Friends - Season 5"
   fetch_file "${emoryNLP_6_URL}" "${emoryNLP_DIR}/friends_season_06.json" "EmoryNLP Friends - Season 6"
   fetch_file "${emoryNLP_7_URL}" "${emoryNLP_DIR}/friends_season_07.json" "EmoryNLP Friends - Season 7"
   fetch_file "${emoryNLP_8_URL}" "${emoryNLP_DIR}/friends_season_08.json" "EmoryNLP Friends - Season 8"
   fetch_file "${emoryNLP_9_URL}" "${emoryNLP_DIR}/friends_season_09.json" "EmoryNLP Friends - Season 9"
   fetch_file "${emoryNLP_10_URL}" "${emoryNLP_DIR}/friends_season_10.json" "EmoryNLP Friends - Season 10"

   # convert original emoryNLP files into TLiDB format w/ 
   #      emotion recognition and reading comprehension datasets
   python3 convert_emoryNLP.py

   # download emoryNLP Question Answering dataset
   fetch_file "${QA_TRAIN_URL}" "emory_question_answering_train.json" 'EmoryNLP Friends Question Answering Dataset - Train'
   fetch_file "${QA_DEV_URL}" "emory_question_answering_dev.json" 'EmoryNLP Friends Question Answering Dataset - Dev'
   fetch_file "${QA_TEST_URL}" "emory_question_answering_test.json" 'EmoryNLP Friends Question Answering Dataset - Test'
   python3 add_friends_QA_annotations.py

   # download emoryNLP Personality Detection dataset
   fetch_file "${PD_URL}" "emory_personality_detection.csv" 'EmoryNLP Friends Personality Detection Dataset'
   python3 add_friends_personality_detection_annotations.py

   # download DialogRE Dataset
   fetch_file "${DIALOGRE_URL}" "dialogre.zip" 'DialogRE Dataset'
   extract_zip 'dialogre.zip' 'DialogRE Dataset'
   python3 add_dialogre_annotations.py

   # download MELD Dataset
   fetch_file "${MELD_TRAIN_URL}" "meld_train.csv" 'MELD Dataset - Train'
   fetch_file "${MELD_DEV_URL}" "meld_dev.csv" 'MELD Dataset - Dev'
   fetch_file "${MELD_TEST_URL}" "meld_test.csv" 'MELD Dataset - Test'
   python3 add_meld_annotations.py

   # add instance ids
   python3 generate_instance_ids.py

   echo "Generating TTiDB splits"
   python3 generate_full_data_TTiDB_ids.py
   echo "Generating Few-Shot TTiDB splits"
   python3 generate_few_shot_data_TTiDB_ids.py

   zip -r TLiDB_Friends.zip TLiDB_Friends
   rm -r "${emoryNLP_DIR}"
   
   files=(
      "emory_question_answering_train.json"
      "emory_question_answering_dev.json"
      "emory_question_answering_test.json"
      "emory_personality_detection.csv"
      "dialogre.zip"
      "dialogre_train_with_map.json"
      "dialogre_dev_with_map.json"
      "dialogre_test_with_map.json"
      "meld_train.csv"
      "meld_dev.csv"
      "meld_test.csv"
   )

   for file in ${files[@]}
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
   local name=$2

   echo "Extracting the ${name} zip"
   unzip "${path}"

   if [[ "$?" -ne 0 ]]; then
      echo "ERROR: Failed to extract ${name} zip"
      exit 40
   fi
}

main "$@"