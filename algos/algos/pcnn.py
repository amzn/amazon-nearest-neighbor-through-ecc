import multiprocessing as mp

# from ..code_db import Code
from ..utils.polar_utils import PolarUtils
from .hashing_algo import HashingAlgo, HashingIndex
from src.algos.polar_codes import PolarCodec
from src.algos.utils.standard_utils import *
import numpy as np
import psutil


class PCNNIndex(HashingIndex):

    def __init__(self, code_dim, code_mask, code_mask_name, **kwargs):
        self.code_dim = code_dim
        self.mask = code_mask
        self.code_mask_name = code_mask_name

        kwargs['bin_dim'] = code_dim * kwargs['num_tables']
        super().__init__(**kwargs)
        self.data_dim = self.num_bits

    def get_params(self):
        index_params = super().get_params()
        index_params['algo_name'] = 'PCNN'
        index_params['code_mask_name'] = self.code_mask_name
        index_params['code_dim'] = self.code_dim
        return index_params

    def get_cluster_ids(self, bin_queries, table_index, num_clusters):
        num_queries = len(bin_queries)
        if num_clusters == 0:
            return np.zeros(shape=(num_queries, 0), dtype=int)
        cluster_ids = np.zeros(shape=(num_queries, num_clusters), dtype=int)
        list_size = PolarUtils.get_list_size_from_wanted(num_clusters)
        list_codec = self.create_list_codec(list_size=list_size)

        start_ind = table_index * self.code_dim
        end_ind = start_ind + self.code_dim
        table_words = index_packed_point(bin_queries, start=start_ind, end=end_ind)
        # unpack_point()
        # calculate batch size
        mem = psutil.virtual_memory()
        total_mem = mem.total
        batch_size = int(total_mem / (mp.cpu_count() * num_clusters * self.code_dim * 500))
        batch_size = max(batch_size, 1)
        decoded_lists = PolarUtils.decode_polar_list(table_words,
                                                     crop=num_clusters,
                                                     list_codec=list_codec,
                                                     batch_size=batch_size)
        cluster_ids[:, :] = self.data_word_to_index(decoded_lists, self.data_dim)
        return cluster_ids

    BASE_LLR_MULT = 3
    LIST_CODEC_CLASS = PolarCodec
    DEFAULT_BATCH_SIZE = 160000

    def get_save_params(self):
        params = super().get_save_params()
        if self.code_mask_name is not None:
            params['code_mask'] = self.mask
            params['code_mask_name'] = self.code_mask_name
        return params

    def load_from_params(self, params):
        rewrite = super().load_from_params(params)
        if 'code_mask' in params:
            self.mask = params['code_mask']
            self.code_mask_name = params['code_mask_name']
        return rewrite

    def create_list_codec(self, list_size):
        return self.LIST_CODEC_CLASS(N=self.code_dim, K=self.data_dim, L=list_size, mask=self.mask)


class PCNN(HashingAlgo):
    INDEX_CLASS = PCNNIndex

    def __init__(self, **kwargs):
        """
        Initializes the PCNN object. The arguments are:
        :key pc_list: the number of codewords to be returned by the list decoder
        :key data_dim: the dimension of the data in the code. Assumption: this is a multiple of 8.
        :key dim: the dimension of the points. Assumption: this is a multiple of 8.
        :key list_codec_class: the class to use for list decoding. If missing, a default class is chosen.
        """
        kwargs['bin_dim'] = kwargs['code_dim'] * kwargs['num_tables']
        kwargs['data_dim'] = kwargs['num_bits']
        super().__init__(**kwargs)

        self.data_dim = kwargs['data_dim']
        self.code_dim = kwargs['code_dim']

        if 'code_mask' in kwargs:
            self.mask = kwargs['code_mask']
            self.code_mask_name = kwargs['code_mask_name']
        else:
            raise Exception('no mask given')

    def get_index_config(self):
        config = super().get_index_config()
        config['code_dim'] = self.code_dim
        config['code_mask'] = self.mask
        config['code_mask_name'] = self.code_mask_name
        return config

    def set_probe_size(self, probe_size):
        self.probe_size = probe_size
