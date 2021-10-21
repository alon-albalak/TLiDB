wget https://github.com/emorynlp/reading-comprehension/raw/master/json/reading-comprehension-trn.json
wget https://github.com/emorynlp/reading-comprehension/raw/master/json/reading-comprehension-dev.json
wget https://github.com/emorynlp/reading-comprehension/raw/master/json/reading-comprehension-tst.json


python3 convert_friends_RC.py

zip -r TLiDB_friends_RC TLiDB_friends_RC/

rm reading-comprehension-trn.json
rm reading-comprehension-dev.json
rm reading-comprehension-tst.json