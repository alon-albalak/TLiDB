wget https://github.com/emorynlp/FriendsQA/raw/master/dat/friendsqa_trn.json
wget https://github.com/emorynlp/FriendsQA/raw/master/dat/friendsqa_dev.json
wget https://github.com/emorynlp/FriendsQA/raw/master/dat/friendsqa_tst.json

python3 convert_friends_QA.py

zip -r TLiDB_friends_QA TLiDB_friends_QA/

rm friendsqa_trn.json
rm friendsqa_dev.json
rm friendsqa_tst.json