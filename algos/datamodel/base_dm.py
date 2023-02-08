from abc import abstractmethod
import os
import numpy as np
import random
from ..saved_object import SavedObject


class BaseDataModel(SavedObject):
    DATAMODEL_PATH = 'Resources/Datasets/'
    QUERY_PATH = 'Resources/Queries/'

    QUERY_CLASS = None

    METRIC_L2 = 0
    METRIC_IP = 1
    METRIC_HAMMING = 2

    METRIC = None

    def __init__(self, *, size_ds, dim):
        super().__init__()
        self.size_ds = size_ds
        self.dim = dim
        self.dataset = None

    def get_dataset(self):
        return self.dataset

    def generate_queries(self, num_queries):
        query_obj = self.QUERY_CLASS(datamodel=self, num_queries=num_queries)
        return query_obj

    ##################### savedobject functions ##########################

    def generate(self):
        super().generate()

    def get_save_params(self):
        """
        returns a dictionary mapping from strings (content names) to arraylike content (ndarray, int, float...).
        This dictionary is the content to be saved.
        :return:
        """
        params = super().get_save_params()
        params['dataset'] = self.dataset
        return params

    def load_from_params(self, params):
        rewrite = False
        if 'id' not in params:
            params = dict(params)
            params['id'] = params['dm_id']
            rewrite = True
        super_rewrite = super().load_from_params(params)
        rewrite = rewrite or super_rewrite
        self.dataset = params['dataset']
        self.dataset.flags.writeable = False
        return rewrite

    ##################### name params ##########################
    @abstractmethod
    def get_params(self):
        return {}


class Query(SavedObject):
    METRIC = None

    def __init__(self, *, num_queries, datamodel):
        self.num_queries = num_queries
        self.datamodel = datamodel
        self.queries = None

    @abstractmethod
    def generate(self):
        super().generate()

    def get_save_params(self):
        params = super().get_save_params()
        params['queries'] = self.queries
        params['dm_id'] = self.datamodel.obj_id
        return params

    def load_from_params(self, params):
        rewrite = False
        if 'id' not in params and 'query_id' in params:
            params = dict(params)
            params['id'] = params['query_id']
            rewrite = True
        super_rewrite = super().load_from_params(params)

        rewrite = rewrite or super_rewrite
        self.queries = params['queries']
        self.queries.flags.writeable = False
        assert self.datamodel.obj_id == params['dm_id']
        return rewrite
