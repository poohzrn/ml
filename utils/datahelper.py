from os import makedirs
from os.path import exists
from pickle import HIGHEST_PROTOCOL, dump, load

IRIS = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
DATA_PATH = 'data/'

import pandas as pd


def save_obj(obj, name):
    "Save an object to disk"
    if not exists(DATA_PATH):
        makedirs(DATA_PATH)
    with open(DATA_PATH + name + '.pkl', 'wb') as _file:
        dump(obj, _file, HIGHEST_PROTOCOL)


def load_obj(name):
    "Loads an object from disk "
    try:
        with open(DATA_PATH + name + '.pkl', 'rb') as _file:
            return load(_file)
    except FileNotFoundError as error:
        print("file {0} not found".format(name))
        load_iris()
        raise error


def load_iris():
    "Loads iris into a dataframe"
    data_name = 'iris.data'
    try:
        return load_obj(data_name)
    except FileNotFoundError as e:
        try:
            df = pd.read_csv(IRIS, header=None)
            save_obj(df, data_name)
            return df
        except Exception as e:
            raise e
