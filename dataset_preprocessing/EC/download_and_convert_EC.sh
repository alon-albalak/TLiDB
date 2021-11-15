#!/bin/bash

readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

readonly TRAIN_DATA_URL='https://emocontextdata.blob.core.windows.net/data/starterkitdata.zip'
readonly TEST_DATA_URL='https://emocontextdata.blob.core.windows.net/data/test.zip'
readonly DATA_DIR="${THIS_DIR}/ec_data"

wget -P ${DATA_DIR} ${TRAIN_DATA_URL}
wget -P ${DATA_DIR} ${TEST_DATA_URL}

for setname in starterkitdata test; do
    unzip ${DATA_DIR}/${setname}.zip -d ${DATA_DIR}/${setname}
done

python3 convert_EC.py

zip -r TLiDB_EC TLiDB_EC/

rm -r ${DATA_DIR}