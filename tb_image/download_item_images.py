#!/usr/bin/env python
# coding=utf-8

"""
Downloading item images from alicdn and save it to oss
"""

import os
import urllib
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

from DLAPP.tb_image.utils import (
    retry,
    type_check,
    logger
)
from DLAPP.tb_image.basic_param import (
    URL_PREFIX,
    URL_FILE,
    IMAGE_PATH,
    CLASS_NAME,
    CLASS_NAME_MAP
)


_RETRIABLE_ERRNOS = {
    110,  # Connection timed out [socket.py]
}


def _is_retriable(e):
    return isinstance(e, IOError) and e.errno in _RETRIABLE_ERRNOS


@retry(initial_delay=1.0, max_delay=16.0, is_retriable=_is_retriable)
def urlretrieve_with_retry(url, filename=None):
    if os.path.exists(filename):
        return None
    return urllib.request.urlretrieve(url, filename)

@type_check
def _download_each_image(line: str) -> int:
    parts = line.strip().split(',')
    if len(parts) != 3:
        logger.warning("Broken record: %s \n", line)
        return -1

    url = parts[1]
    class_id = CLASS_NAME_MAP[parts[2]]
    if url is None or len(url) == 0:
        return -1

    urlretrieve_with_retry(os.path.join(URL_PREFIX, url), os.path.join(IMAGE_PATH, class_id, url))
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

    for line in lines:
        _download_each_image(line)

    # thread parallel
#    pool = ThreadPool(10)
#    res = pool.map(_download_each_image, lines)
#    pool.close()

#    pool.join()

    # download image using multi-processing
#    pool = Pool(10)
#    logger.info("Downloading image using %d processes\n", pool._processes)
#    try :
#        res = pool.map(_download_each_image, lines)
#    except (IOError) as e:
#        print(e)
#    pool.close()

if __name__ == '__main__':
    maybe_download()