"""
Downloading item images from alicdn and save it to oss
"""

import os
import urllib.request
import logging
from multiprocessing import Pool

from DLAPP.tb_image.utils import retry, type_check

from tensorflow.python.platform import gfile
import tensorflow as tf

URL_FILE = "/Users/cheng/Data/data/nvzh_item_style_image_samples"
URL_PREFIX = "https://img.alicdn.com/imgextra/"
IMAGE_PATH = "/Users/cheng/Data/data/item_images"
CLASS_NAME = ['1', '2', '3']


_RETRIABLE_ERRNOS = {
    110,  # Connection timed out [socket.py]
}


def _is_retriable(e):
    return isinstance(e, IOError) and e.errno in _RETRIABLE_ERRNOS


@retry(initial_delay=1.0, max_delay=16.0, is_retriable=_is_retriable)
def urlretrieve_with_retry(url, filename=None):
    logging.info(url)
    return urllib.request.urlretrieve(url, filename)


@type_check
def _download_each_image(line: str) -> int:
    parts = line.strip().split(',')
    if len(parts) != 3:
        logging.warning("Broken record: %s \n", line)
        return -1
    url = parts[1]
    class_id = parts[2]
    urlretrieve_with_retry(os.path.join(URL_PREFIX, url), os.path.join(IMAGE_PATH, class_id.strip(), url))
    return 0


def maybe_download():
    if not os.path.exists(IMAGE_PATH):
        os.mkdir(IMAGE_PATH)
    for class_name in CLASS_NAME:
        path = os.path.join(IMAGE_PATH, class_name)
        if not os.path.exists(path):
            os.mkdir(path)

    with open(URL_FILE, mode='r') as f:
        lines = f.readlines()

    # download image using multi-processing
    #for line in lines[:5]:
    #    _download_each_image(line)

    pool = Pool()
    logging.info("Downloading image using %d processes\n", pool._processes)
    res = pool.map(_download_each_image, lines[:1])
    pool.close()


if __name__ == '__main__':
    maybe_download()