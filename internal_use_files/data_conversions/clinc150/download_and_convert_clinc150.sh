wget -O original_clinc150.json https://github.com/clinc/oos-eval/raw/master/data/data_full.json
wget -O clinc150_domains.json https://github.com/clinc/oos-eval/raw/master/data/domains.json

python3 convert_clinc150.py

zip -r TLiDB_clinc150 TLiDB_clinc150/

rm original_clinc150.json
rm clinc150_domains.json