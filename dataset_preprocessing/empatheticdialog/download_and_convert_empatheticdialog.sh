DOWNLOAD_FOLDER=original_empatheticdialog

wget -P ${DOWNLOAD_FOLDER} wget https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz
tar -zxvf ${DOWNLOAD_FOLDER}/empatheticdialogues.tar.gz -C ${DOWNLOAD_FOLDER} --strip-components=1

python3 convert_empatheticdialog.py

zip -r TLiDB_empatheticdialog TLiDB_empatheticdialog/

rm -r $DOWNLOAD_FOLDER
