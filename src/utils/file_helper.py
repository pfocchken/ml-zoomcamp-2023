"""Module contains helper methods to work with file"""
import os
from os.path import isfile
import wget


DESTINATION_FOLDER = "../etc"


def get_dataset_file(file_url: str, file_name: str) -> str:
    """Get file from remote url and save it to destination_file

    :param file_url: url from where file should be downloaded
    :param file_name: name where file should be saved
    :return: path to downloaded file
    """
    file_destination = os.path.join(os.getcwd(), DESTINATION_FOLDER, file_name)

    if not isfile(file_destination):
        wget.download(file_url, out=file_destination)

    return file_destination

