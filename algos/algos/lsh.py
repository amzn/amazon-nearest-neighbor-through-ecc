from .hashing_algo import HashingAlgo, HashingIndex
import multiprocessing as mp
import psutil
from src.algos.utils.standard_utils import *

from itertools import chain, combinations


class LSHIndex(HashingIndex):

    def __init__(self, **kwargs):
        kwargs['bin_dim'] = kwargs['num_bits'] * kwargs['num_tables']
        super().__init__(**kwargs)

    def get_hash(self, bin_words, table_ind):
        mem = psutil.virtual_memory()
        total_mem = mem.total

        # holding the bin_words in memory should not take more than some constant fraction of the memory share of the process.
        batch_size = int(total_mem / (mp.cpu_count() * self.bin_dim * 100))
        batch_size = max(batch_size, 1)

        proc_result = np.zeros(shape=len(bin_words), dtype=int)
        for i in range(0, len(bin_words), batch_size):
            batch_start = i
            batch_end = min(len(bin_words), i + batch_size)
            hash_words = self.near_compliant_hash(bin_words[batch_start:batch_end], table_ind)

            proc_result[batch_start:batch_end] = hash_words
        return proc_result

    def near_compliant_hash(self, words, table_ind):
        start_bit_ind = table_ind * self.num_bits
        end_bit_ind = start_bit_ind + self.num_bits
        ext_word = index_packed_point(words, start_bit_ind, end_bit_ind)
        return self.data_word_to_index(ext_word, self.num_bits)

    def get_cluster_ids(self, bin_queries, table_index, num_clusters):
        num_queries = len(bin_queries)
        cluster_ids = np.zeros(shape=(num_queries, num_clusters), dtype=int)
        flip_masks = self._get_flip_masks(num_clusters)
        hashes = self.get_hash(bin_queries, table_index)
        cluster_ids[:, :] = self._get_flips(hashes, flip_masks)
        return cluster_ids

    def _get_flips(self, words, flip_masks):
        return words[:, np.newaxis] ^ flip_masks[np.newaxis, :]

    def _get_flip_masks(self, num_flips):
        if num_flips == 0:
            return np.ndarray(0, dtype=int)

        # generate flip numbers
        set_list = []
        ind_list = list(range(self.num_bits))
        num_added = 0
        for val in chain.from_iterable(combinations(ind_list, r) for r in range(len(ind_list) + 1)):
            set_list.append(val)
            num_added += 1
            if num_added == num_flips:
                break
            if num_added > num_flips:
                raise Exception('what.')

        def set_to_mask(bit_set):
            result = 0
            for bit in bit_set:
                result ^= (1 << bit)
            return result

        return np.array([set_to_mask(bit_set) for bit_set in set_list], dtype=int)

    def get_params(self):
        index_params = super().get_params()
        index_params['algo_name'] = 'NoamLSH'
        return index_params


class LSH(HashingAlgo):
    INDEX_CLASS = LSHIndex

    def __init__(self, **kwargs):
        kwargs['bin_dim'] = kwargs['num_tables'] * kwargs['num_bits']
        super().__init__(**kwargs)

    def get_index_config(self):
        return super().get_index_config()

    def set_probe_size(self, probe_size):
        self.probe_size = probe_size
