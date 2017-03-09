#encoding: utf-8

import os
import re

def load_file_list(path=None, regx=None, printable=True):
    """Return a file list in a folder by given a path and regular expression.

    Parameters
    ----------
    path : a string or None
        A folder path.
    regx : a string
        The regx of file name.
    printable : boolean, whether to print the files infomation.
    """
    if path is None:
        path = os.getcwd()
    file_list = os.listdir(path)
    return_list = []

    if regx is None:
        return_list =  file_list
    else:
        for idx, f in enumerate(file_list):
            if re.search(regx, f):
                return_list.append(f)
    # return_list.sort()
    if printable:
        print('Match file list = %s' % return_list)
        print('Number of files = %d' % len(return_list))
    return return_list

if __name__ == '__main__':
    load_file_list()
