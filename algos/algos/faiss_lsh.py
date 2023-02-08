import os

import faiss
import numpy as np

from .algo import Algo, Index
from ..datamodel import BaseDataModel


class FaissLSHIndex(Index):

    def __init__(self, *, num_bits, num_tables, **kwargs):
        super().__init__(**kwargs)
        self.num_bits = num_bits
        self.num_tables = num_tables
        self.num_flips = None

    def get_params(self):
        return {'algo_name': 'FaissLSH',
                'num_bits': self.num_bits,
                'num_tables': self.num_tables}

    def generate(self):
        super().generate()
        self.faiss_index = faiss.IndexBinaryMultiHash(self.datamodel.dim, self.num_tables, self.num_bits)
        dataset = self.datamodel.get_dataset()
        self.faiss_index.add(dataset)

    def get_save_params(self) -> dict:
        save_params = super().get_save_params()
        if self.datamodel.METRIC == BaseDataModel.METRIC_HAMMING:
            chunk = faiss.serialize_index_binary(self.faiss_index)
        else:
            raise Exception("faiss only supports LSH clustering for binary datasets.")
        save_params['faiss_index'] = chunk
        return save_params

    def load_from_params(self, params):
        rewrite = super().load_from_params(params)
        if self.datamodel.METRIC == BaseDataModel.METRIC_HAMMING:
            self.faiss_index = faiss.deserialize_index_binary(params['faiss_index'])
        else:
            raise Exception("faiss only supports LSH clustering for binary datasets.")
        return rewrite

    def set_n_probe(self, n_probe):
        self.faiss_index.num_flips = n_probe

    def search(self, queries, size_nn):
        distances, points = self.faiss_index.search(queries, size_nn)
        return distances, points

    def set_num_flips(self, num_flips):
        self.num_flips = num_flips
        self.faiss_index.nflip = num_flips


class FaissLSH(Algo):
    INDEX_CLASS = FaissLSHIndex

    """
    This is the benchmark NN algorithm, based on FAISS's IndexBinaryIVF class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.num_bits = kwargs['num_bits']
        self.num_tables = kwargs['num_tables']
        self.num_flips = None
        self.num_queries = 0

    def get_index_config(self):
        return {
            'num_bits': self.num_bits,
            'num_tables': self.num_tables
        }

    def query(self, datamodel, query_obj, size_nn):
        self.zero_stats()
        queries = query_obj.queries
        distances, points = self.index_obj.search(queries, size_nn)
        self.num_queries += len(queries)
        return distances, points

    def set_num_flips(self, num_flips):
        self.num_flips = num_flips
        self.index_obj.set_num_flips(num_flips)

    def get_ndis(self):
        return faiss.cvar.indexBinaryHash_stats.ndis

    def zero_stats(self):
        stats = faiss.cvar.indexBinaryHash_stats
        stats.ndis = 0
        stats.n0 = 0
        stats.nlist = 0
        self.num_queries = 0

    def get_num_queries(self):
        return self.num_queries

    def get_nop(self):
        stats = faiss.cvar.indexBinaryHash_stats
        return stats.ndis + stats.nlist + stats.n0

    def get_probing_params(self):
        return {'num_flips': self.num_flips}
