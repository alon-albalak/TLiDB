DOWNLOAD_FOLDER=original_multiwoz22

wget -P ${DOWNLOAD_FOLDER}/test https://raw.githubusercontent.com/budzianowski/multiwoz/master/data/MultiWOZ_2.2/test/dialogues_001.json
wget -P ${DOWNLOAD_FOLDER}/test https://raw.githubusercontent.com/budzianowski/multiwoz/master/data/MultiWOZ_2.2/test/dialogues_002.json

wget -P ${DOWNLOAD_FOLDER}/dev https://raw.githubusercontent.com/budzianowski/multiwoz/master/data/MultiWOZ_2.2/dev/dialogues_001.json
wget -P ${DOWNLOAD_FOLDER}/dev https://raw.githubusercontent.com/budzianowski/multiwoz/master/data/MultiWOZ_2.2/dev/dialogues_002.json

wget -P ${DOWNLOAD_FOLDER}/train https://raw.githubusercontent.com/budzianowski/multiwoz/master/data/MultiWOZ_2.2/train/dialogues_001.json
wget -P ${DOWNLOAD_FOLDER}/train https://raw.githubusercontent.com/budzianowski/multiwoz/master/data/MultiWOZ_2.2/train/dialogues_002.json
wget -P ${DOWNLOAD_FOLDER}/train https://raw.githubusercontent.com/budzianowski/multiwoz/master/data/MultiWOZ_2.2/train/dialogues_003.json
wget -P ${DOWNLOAD_FOLDER}/train https://raw.githubusercontent.com/budzianowski/multiwoz/master/data/MultiWOZ_2.2/train/dialogues_004.json
wget -P ${DOWNLOAD_FOLDER}/train https://raw.githubusercontent.com/budzianowski/multiwoz/master/data/MultiWOZ_2.2/train/dialogues_005.json
wget -P ${DOWNLOAD_FOLDER}/train https://raw.githubusercontent.com/budzianowski/multiwoz/master/data/MultiWOZ_2.2/train/dialogues_006.json
wget -P ${DOWNLOAD_FOLDER}/train https://raw.githubusercontent.com/budzianowski/multiwoz/master/data/MultiWOZ_2.2/train/dialogues_007.json
wget -P ${DOWNLOAD_FOLDER}/train https://raw.githubusercontent.com/budzianowski/multiwoz/master/data/MultiWOZ_2.2/train/dialogues_008.json
wget -P ${DOWNLOAD_FOLDER}/train https://raw.githubusercontent.com/budzianowski/multiwoz/master/data/MultiWOZ_2.2/train/dialogues_009.json
wget -P ${DOWNLOAD_FOLDER}/train https://raw.githubusercontent.com/budzianowski/multiwoz/master/data/MultiWOZ_2.2/train/dialogues_010.json
wget -P ${DOWNLOAD_FOLDER}/train https://raw.githubusercontent.com/budzianowski/multiwoz/master/data/MultiWOZ_2.2/train/dialogues_011.json
wget -P ${DOWNLOAD_FOLDER}/train https://raw.githubusercontent.com/budzianowski/multiwoz/master/data/MultiWOZ_2.2/train/dialogues_012.json
wget -P ${DOWNLOAD_FOLDER}/train https://raw.githubusercontent.com/budzianowski/multiwoz/master/data/MultiWOZ_2.2/train/dialogues_013.json
wget -P ${DOWNLOAD_FOLDER}/train https://raw.githubusercontent.com/budzianowski/multiwoz/master/data/MultiWOZ_2.2/train/dialogues_014.json
wget -P ${DOWNLOAD_FOLDER}/train https://raw.githubusercontent.com/budzianowski/multiwoz/master/data/MultiWOZ_2.2/train/dialogues_015.json
wget -P ${DOWNLOAD_FOLDER}/train https://raw.githubusercontent.com/budzianowski/multiwoz/master/data/MultiWOZ_2.2/train/dialogues_016.json
wget -P ${DOWNLOAD_FOLDER}/train https://raw.githubusercontent.com/budzianowski/multiwoz/master/data/MultiWOZ_2.2/train/dialogues_017.json


python3 convert_multiwoz22.py

rm -r $DOWNLOAD_FOLDER