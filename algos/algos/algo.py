from abc import ABC, abstractmethod
import numpy as np
import os

from src.algos.saved_object import SavedObject


class Algo(ABC):
    """
    This is a abstract base class for all NN algorithms.
    """

    INDEX_PATH = 'Resources/Indices/'
    OUTPUT_PATH = 'Resources/Outputs/'

    INDEX_CLASS = None

    def __init__(self, dim, **kwargs):
        self.dim = dim
        self.index_obj = None

    def generate_index(self, datamodel):
        """
        Generates an index for the given datamodel (based on the index type for the specific class).
        """
        index_config = self.get_index_config()
        self.index_obj = self.INDEX_CLASS(datamodel=datamodel,
                                          **index_config)
        self.index_obj.generate()

    def load_index(self, datamodel, filepath):
        """
        This function loads the given algo index for "datamodel" from "filepath"
        """
        index_config = self.get_index_config()
        self.index_obj = self.INDEX_CLASS(datamodel=datamodel,
                                          **index_config)
        self.index_obj.load(filepath)

    def save_index(self, filepath):
        self.index_obj.save(filepath)

    def set_index(self, index_obj):
        self.index_obj = index_obj

    @abstractmethod
    def query(self, datamodel, query_obj, size_nn: int):
        pass

    def get_index_params(self):
        return self.index_obj.get_params()

    @abstractmethod
    def get_probing_params(self):
        return {}

    def get_params(self):
        """
        Returns the parameters defining the algorithm. This includes both the parameters of the index and the parameters of the probing technique.
        """
        params = self.get_index_params()
        params.update(self.get_probing_params())
        return params

    @classmethod
    def name_from_params(cls, params):
        name = params['algo_name']
        params_no_name = dict(params)
        del params_no_name['algo_name']
        param_list = [key + '=' + str(value) for key, value in params_no_name.items()]
        param_string = ', '.join(param_list)
        if param_string == '':
            return f'{name}'
        else:
            return f'{name}: {param_string}'

    @abstractmethod
    def get_index_config(self):
        """
        Returns the parameters to be given to the index constructor, to be used for generation/loading.
        """
        pass

    '''stats methods'''

    @abstractmethod
    def zero_stats(self):
        pass

    @abstractmethod
    def get_ndis(self):
        """
        Returns the number of distance query_dict from the time of the algo's creation (or from the last zero_ndis
        function call).
        """
        pass

    @abstractmethod
    def get_num_queries(self):
        pass

    @abstractmethod
    def get_nop(self):
        pass


class Index(SavedObject):
    def __init__(self, datamodel, **kwargs):
        super().__init__(**kwargs)
        self.datamodel = datamodel

    @abstractmethod
    def get_params(self):
        """
        Returns a dictionary of parameters that characterize the index.
        """
        return {}

    def load_from_params(self, params):
        rewrite = super().load_from_params(params)
        assert params['dm_id'] == self.datamodel.obj_id
        return rewrite

    def get_save_params(self) -> dict:
        save_params = super().get_save_params()
        save_params['dm_id'] = self.datamodel.obj_id
        return save_params
