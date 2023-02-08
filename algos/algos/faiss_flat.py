import faiss
import numpy as np

from .algo import Algo
from ..datamodel import BaseDataModel


class ExactSearch(Algo):
    """
    This is the bruteforce algorithm for NN, used to provide ground truths.
    It is implemented using FAISS's IndexBinaryFlat class.
    """

    INDEX_CLASS = None

    def get_probing_params(self):
        return {}

    def get_index_config(self):
        raise Exception('exact search does not have index')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.index = None

    def create_index(self, datamodel):
        if datamodel.METRIC == BaseDataModel.METRIC_HAMMING:
            self.index = faiss.IndexBinaryFlat(self.dim)
        elif datamodel.METRIC == BaseDataModel.METRIC_L2:
            self.index = faiss.IndexFlatL2(self.dim)
        elif datamodel.METRIC == BaseDataModel.METRIC_IP:
            self.index = faiss.IndexFlatIP(self.dim)
        else:
            raise Exception('unimplemented')

        self.index.add(datamodel.get_dataset())

    def query(self, *, datamodel, query_obj, size_nn):
        D, I = self.index.search(query_obj.queries, size_nn)
        if datamodel.METRIC == BaseDataModel.METRIC_L2:
            # add square root to l2 squared
            D = np.sqrt(D)
        elif datamodel.METRIC == BaseDataModel.METRIC_IP:
            # Turn cosine similarity to cosine difference
            D = 1 - D

        return D, I

    def get_ndis(self):
        raise Exception("ExactSearch: get_ndis not implemented")

    def get_nop(self):
        raise Exception("ExactSearch: get_nop not implemented")

    def get_num_queries(self):
        raise Exception("ExactSearch: get_num_queries not implemented")

    def zero_stats(self):
        raise Exception("ExactSearch: zero_stats not implemented")

    def get_params(self):
        return {'algo_name': 'FaissFlat'}
