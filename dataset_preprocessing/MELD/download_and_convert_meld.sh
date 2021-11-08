#!/bin/bash

readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

readonly DATA_URL='http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz'
readonly DATA_FILE="${THIS_DIR}/MELD.RAW.tar.gz"
readonly DATA_DIR="${THIS_DIR}/MELD.RAW"

function main() {
   trap exit SIGINT

   check_requirements
   echo "requirements satisfied"
   fetch_file "${DATA_URL}" "${DATA_FILE}" 'data'

   echo "fetch completed"
   extract_zip "${DATA_FILE}" "${DATA_DIR}" 'data'

   python3 convert_meld.py

   zip -r TLiDB_MELD.zip TLiDB_MELD/
   rm -r ${DATA_DIR}
   rm ${DATA_FILE}
   rm -r train_splits
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

function fetch_file() {
   local url=$1
   local path=$2
   local name=$3

   if [[ -e "${path}" ]]; then
      echo "${name} file found cached, skipping download."
      return
   fi

   echo "Downloading ${name} file with command: $FETCH_COMMAND"
#   `get_fetch_command` "${path}" "${url}"

  wget "${path}" "${url}"
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

   echo "Extracting the ${path} zip"
   tar -xf "${path}"

   tar -xf "${expectedDir}/train.tar.gz"
   mv "train_sent_emo.csv" "${expectedDir}/train_sent_emo.csv"

   if [[ "$?" -ne 0 ]]; then
      echo "ERROR: Failed to extract ${path} tar.gz"
      exit 40
   fi
}


main "$@"