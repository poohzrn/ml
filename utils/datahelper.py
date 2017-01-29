from os import makedirs
from os.path import exists
from pickle import HIGHEST_PROTOCOL, dump, load

import pandas as pd

IRIS = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'


class DataHandler:
    "wrapper for handling files"

    data_path = 'data/'

    def __init__(self):
        if not exists(self.data_path):
            makedirs(self.data_path)

    @classmethod
    def save_obj(cls, obj, name):
        """Sav an object to disk

        :obj: TODO
        :name: TODO
        :returns: TODO

        """
        with open(cls.data_path + name + '.pkl', 'wb') as _file:
            dump(obj, _file, HIGHEST_PROTOCOL)

    @classmethod
    def load_obj(cls, name):
        """Loads an object from disk

        :name: TODO
        :returns: TODO

        """
        try:
            with open(cls.data_path + name + '.pkl', 'rb') as _file:
                return load(_file)
        except FileNotFoundError as error:
            print("file {0} not found".format(name))
            print(error)
            raise error

    @classmethod
    def load_iris(cls):
        """Loads iris into a dataframe
        :returns: TODO

        """
        data_name = 'iris.data'
        try:
            return cls.load_obj(data_name)
        except FileNotFoundError as e:
            try:
                df = pd.read_csv(IRIS, header=None)
                cls.save_obj(df, data_name)
                return df
            except Exception as e:
                raise e
