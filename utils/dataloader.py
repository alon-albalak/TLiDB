import os
import json
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
import logging
logger = logging.getLogger(__name__)

# loads dataset into memory, caches them if they are not already on disk

# dataset url can be found in the google drive, where the original link is:
#   https://drive.google.com/file/d/1sqaiYTm9b9SPEzehdjp_DXEovVId6Fvq/view?usp=sharing
# and needs to be reformatted as:
#   https://drive.google.com/uc?export=download&id=1sqaiYTm9b9SPEzehdjp_DXEovVId6Fvq
dataset_urls = {
    "multiwoz22": "https://drive.google.com/uc?export=download&id=1sqaiYTm9b9SPEzehdjp_DXEovVId6Fvq",
    "clinc150": "https://drive.google.com/uc?export=download&id=1Vu_1GjGF3bJo77Amw5qNU7z8Smn2B6Av",
    "friends_ER": "https://drive.google.com/uc?export=download&id=13b8PsYGShUDkoYS6m4YHbqWlxaOW3i_L",
    "friends_RC": "https://drive.google.com/uc?export=download&id=1QBgOV6YI8wuQpRAu0XPxzoM77ma7zA_o",
    "friends_QA": "https://drive.google.com/uc?export=download&id=1rhR-l-9Zl9cUpCOSpHbsBedfBUmTRVbN"
}

def download_and_unzip(url, extract_to='.'):
    logger.info(f"Waiting for response from {url}")
    http_response = urlopen(url)
    logger.info("Downloading data from {url}")
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)

def load_dataset_local(name):
    ds = {}
    for root, dirs, files in os.walk(f"datasets/TLiDB_{name}"):
        for file in files:
            if file.endswith(".json") and file!="sample_format.json":
                ds[file[:-5]] = json.load(open(f"{root}/{file}"))
    return ds

def load_dataset(name):
    assert(name in dataset_urls.keys()), f"Could not find {name} in the datasets\
                \nTry using any of {' '.join(list(dataset_urls.keys()))}"

    # download and unzip dataset if needed
    if f"TLiDB_{name}" not in os.listdir("datasets"):
        download_and_unzip(dataset_urls[name],"datasets")
        logger.info(f"Extracted files to /datasets/{name}")

    ds = load_dataset_local(name)

    return ds