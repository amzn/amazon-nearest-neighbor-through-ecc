import faiss
import numpy as np

from src.algos import Algo, BaseDataModel


class HNSW(Algo):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hnsw_m = kwargs['hnsw_m']
        self.ef_construction = kwargs['ef_construction']

        # the number of query_dict since the last ndis zeroing, needed for accurate ndis.
        self.num_queries = 0

    def get_index_params(self):
        return {'algo_name': 'HNSW',
                'hnsw_m': self.hnsw_m,
                'ef_construction': self.ef_construction}

    def get_probing_params(self):
        return {'ef_search': self.ef_search}

    def create_index(self, datamodel):

        if datamodel.METRIC == BaseDataModel.METRIC_L2:
            self.index = faiss.IndexHNSWFlat(self.dim, self.hnsw_m)
        elif datamodel.METRIC == BaseDataModel.METRIC_IP:
            self.index = faiss.IndexHNSWFlat(self.dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        else:
            raise Exception('unimplemented')

        self.index.hnsw.efConstruction = self.ef_construction

        self.ef_search = None

        dataset = datamodel.get_dataset()
        self.index.train(dataset)
        self.index.add(dataset)

    def load_index(self, *, datamodel, dm_filename):
        index_filename = self.get_index_filename(dm_filename=dm_filename)
        file_cont = np.load(self.INDEX_PATH + index_filename)
        self.index = faiss.deserialize_index(file_cont['faiss_index'])

    def save_index(self, dm_filename):
        filename = self.get_index_filename(dm_filename=dm_filename)
        filepath = self.INDEX_PATH + filename
        chunk = faiss.serialize_index(self.index)
        np.savez(filepath, faiss_index=chunk)
        return True

    def query(self, datamodel, query_dict, size_nn):
        self.zero_stats()
        queries = query_dict['queries']
        distances, n_ids = self.index.search(queries, size_nn)
        if datamodel.METRIC == BaseDataModel.METRIC_L2:
            # adding square root
            distances = np.sqrt(distances)
        elif datamodel.METRIC == BaseDataModel.METRIC_IP:
            # Turn cosine similarity to cosine difference
            distances = 1 - distances

        self.num_queries += len(queries)
        return distances, n_ids

    def set_ef_search(self, ef_search):
        self.ef_search = ef_search
        self.index.hnsw.efSearch = ef_search

    def get_ndis(self):
        ndis = faiss.cvar.hnsw_stats.n3
        return ndis

    def get_nop(self):
        return self.get_ndis()

    def get_num_queries(self):
        return self.num_queries

    def zero_stats(self):
        faiss.cvar.hnsw_stats.reset()
        self.num_queries = 0
