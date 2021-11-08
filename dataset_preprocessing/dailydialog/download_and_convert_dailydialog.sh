DOWNLOAD_FOLDER=original_dailydialog

wget -P ${DOWNLOAD_FOLDER} http://yanran.li/files/ijcnlp_dailydialog.zip
unzip ${DOWNLOAD_FOLDER}/ijcnlp_dailydialog.zip -d ${DOWNLOAD_FOLDER}

FILE_FOLDER=${DOWNLOAD_FOLDER}/ijcnlp_dailydialog
for setname in train validation test; do
    unzip ${FILE_FOLDER}/${setname}.zip -d ${FILE_FOLDER}
done


python3 convert_dailydialog.py

zip -r TLiDB_dailydialog TLiDB_dailydialog/

rm -r $DOWNLOAD_FOLDER
