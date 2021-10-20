import os
import json
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile

# loads dataset into memory, caches them if they are not already on disk

# dataset url can be found in the google drive, where the original link is:
#   https://drive.google.com/file/d/1sqaiYTm9b9SPEzehdjp_DXEovVId6Fvq/view?usp=sharing
# and needs to be reformatted as:
#   https://drive.google.com/uc?export=download&id=1sqaiYTm9b9SPEzehdjp_DXEovVId6Fvq
dataset_urls = {
    "multiwoz_22": "https://drive.google.com/uc?export=download&id=1sqaiYTm9b9SPEzehdjp_DXEovVId6Fvq",
    "clinc150": "https://drive.google.com/uc?export=download&id=1Vu_1GjGF3bJo77Amw5qNU7z8Smn2B6Av"
}

def download_and_unzip(url, extract_to='.'):
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)

def load_dataset_local(name):
    pass

def load_dataset(name):
    assert(name in dataset_urls.keys()), f"Could not find {name} in the datasets\
                \nTry using any of {' '.join(list(dataset_urls.keys()))}"

    # download and unzip dataset if needed
    if name not in os.listdir("../datasets"):
        download_and_unzip(dataset_urls[name],"../datasets")

    ds = load_dataset_local(name)

    return ds

load_dataset("clinc150")