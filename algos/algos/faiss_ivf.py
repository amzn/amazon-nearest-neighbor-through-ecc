import os

import faiss
import numpy as np

from .algo import Algo, Index
from ..datamodel import BaseDataModel


class FaissIVFIndex(Index):

    def __init__(self, *, datamodel, n_list):
        super().__init__()
        self.n_list = n_list
        self.datamodel = datamodel
        self.n_probe = None

    def get_params(self):
        return {'algo_name': 'FaissIVF',
                'n_list': self.n_list}

    def generate(self):
        super().generate()
        if self.datamodel.METRIC == BaseDataModel.METRIC_L2 \
                or self.datamodel.METRIC == BaseDataModel.METRIC_IP:
            quantizer = faiss.IndexFlatL2(self.datamodel.dim)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, self.datamodel.dim, self.n_list)
        # elif self.datamodel.DATA_TYPE == DataModel.TYPE_REAL_IP:
        #     quantizer = faiss.IndexFlatIP(self.datamodel.dim)
        #     self.faiss_index = faiss.IndexIVFFlat(quantizer, self.datamodel.dim, self.n_list)
        elif self.datamodel.METRIC == BaseDataModel.METRIC_HAMMING:
            quantizer = faiss.IndexBinaryFlat(self.datamodel.dim)
            self.faiss_index = faiss.IndexBinaryIVF(quantizer, self.datamodel.dim, self.n_list)
        else:
            raise Exception('unimplemented distance')
        dataset = self.datamodel.get_dataset()
        self.faiss_index.train(dataset)
        self.faiss_index.add(dataset)

    def get_save_params(self) -> dict:
        save_params = super().get_save_params()
        if self.datamodel.METRIC != BaseDataModel.METRIC_HAMMING:
            chunk = faiss.serialize_index(self.faiss_index)
        else:
            chunk = faiss.serialize_index_binary(self.faiss_index)
        save_params['faiss_index'] = chunk
        return save_params

    def load_from_params(self, params):
        rewrite = super().load_from_params(params)
        if self.datamodel.METRIC != BaseDataModel.METRIC_HAMMING:
            self.faiss_index = faiss.deserialize_index(params['faiss_index'])
        else:
            self.faiss_index = faiss.deserialize_index_binary(params['faiss_index'])
        return rewrite

    def set_n_probe(self, n_probe):
        self.faiss_index.nprobe = n_probe

    def search(self, queries, size_nn):
        distances, points = self.faiss_index.search(queries, size_nn)
        if self.datamodel.METRIC == BaseDataModel.METRIC_L2:
            # add square root to l2 squared
            distances = np.sqrt(distances)
        elif self.datamodel.METRIC == BaseDataModel.METRIC_IP:
            # turn squared L2 into cosine distance
            distances = distances / 2

        return distances, points


class FaissIVF(Algo):
    INDEX_CLASS = FaissIVFIndex

    """
    This is the benchmark NN algorithm, based on FAISS's IndexBinaryIVF class.
    """

    def get_probing_params(self):
        return {'n_probe': self.n_probe}

    def __init__(self, **kwargs):
        """
        Initializes the FaissIVF object to be used. The arguments are:
        :key n_list: the number of clusters to create
        :key dim: the dimension of the dataset (bits).
        """
        super().__init__(**kwargs)
        self.n_list = kwargs['n_list']
        self.n_probe = None

        # the number of query_dict since the last ndis zeroing, needed for accurate ndis.
        self.num_queries = 0

    def get_index_config(self):
        return {
            'n_list': self.n_list
        }

    def query(self, datamodel, query_obj, size_nn):
        self.zero_stats()
        queries = query_obj.queries
        distances, points = self.index_obj.search(queries, size_nn)

        self.num_queries += len(queries)
        return distances, points

    def set_n_probe(self, n_probe):
        self.n_probe = n_probe
        self.index_obj.set_n_probe(n_probe)

    def get_ndis(self):
        ndis = faiss.cvar.indexIVF_stats.ndis
        # the ndis kept by FaissIVF does not take into account cluster comparisons.
        # fix this by adding n_list * num_queries.
        ndis += self.n_list * self.num_queries
        return ndis

    def get_nop(self):
        return self.get_ndis()

    def get_num_queries(self):
        return self.num_queries

    def zero_stats(self):
        faiss.cvar.indexIVF_stats.ndis = 0
        self.num_queries = 0
