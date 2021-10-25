wget https://raw.githubusercontent.com/emorynlp/emotion-detection/master/json/emotion-detection-trn.json
wget https://raw.githubusercontent.com/emorynlp/emotion-detection/master/json/emotion-detection-dev.json
wget https://raw.githubusercontent.com/emorynlp/emotion-detection/master/json/emotion-detection-tst.json


python3 convert_friends_ER.py

zip -r TLiDB_friends_ER TLiDB_friends_ER/

rm emotion-detection-trn.json
rm emotion-detection-dev.json
rm emotion-detection-tst.json