import os, codecs
from utils import RawData

class Reader:
    def __init__(self):
        pass

    def read(dataset_folder_path, get_layout_data=False):
        labels = None
        text_arr = None
        tags_arr = None
        coordinates = None

        with open(os.path.join(dataset_folder_path, 'labels.txt'), encoding='utf-8') as f:
            labels = f.readlines()

        with codecs.open(os.path.join(dataset_folder_path, 'seq_in.txt'), encoding='utf-8', errors='ignore') as f:
            text_arr = f.readlines()

        with open(os.path.join(dataset_folder_path, 'seq_out.txt'), encoding='utf-8') as f:
            tags_arr = f.readlines()

        coordinates = []
        if get_layout_data:
            with open(os.path.join(dataset_folder_path, 'coordinates.txt'), encoding='utf-8') as f:
                coordinates = f.read()
            coordinates = [[[float(x) for x in coord.split()] for coord in coords.split('\n')] for coords in coordinates.split('\n---------\n')]

        labels = [x.strip() for x in labels]
        text_arr = [x.strip() for x in text_arr]
        tags_arr = [x.strip() for x in tags_arr]

        data = RawData(text_arr, tags_arr, coordinates, labels)

        if get_layout_data:
            assert len(text_arr) == len(tags_arr) == len(labels) == len(coordinates) # test by using 'assert'
        assert len(text_arr) == len(tags_arr) == len(labels) # test by using 'assert'
        
        return data