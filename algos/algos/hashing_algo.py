import sys

import cProfile

from datetime import datetime

from .algo import Algo, Index
import multiprocessing as mp
from multiprocessing import sharedctypes
from src.algos.utils.standard_utils import *
from ..datamodel import BaseDataModel


class HashingIndex(Index):
    EMB_PREFIX = 'hpemb_'

    def __init__(self, *, num_bits,
                 num_tables,
                 bin_dim,
                 seed=None,
                 embedding_class='hp',
                 **kwargs):
        super().__init__(**kwargs)
        self.num_bits = num_bits
        self.num_tables = num_tables
        self.bin_dim = bin_dim
        self.seed = seed
        self.index = None
        self.dataset_indices = None
        self.needs_embedding = (self.datamodel.METRIC != BaseDataModel.METRIC_HAMMING)
        if self.needs_embedding:
            self.embedding_class = embedding_class
            if self.embedding_class == 'hp':
                self.embedding = HyperplaneEmbedding(source_dim=self.datamodel.dim, dest_dim=self.bin_dim, seed=seed)
            elif self.embedding_class == 'shp':
                self.prrot_iters = kwargs['prrot_iters']
                self.embedding = SpeedyHyperplaneEmbedding(source_dim=self.datamodel.dim, dest_dim=self.bin_dim,
                                                           seed=seed, prrot_iters=self.prrot_iters)
            elif self.embedding_class == 'ssrae':
                self.embedding = AutoencoderEmbedding(size_ds=self.datamodel.size_ds,
                                                      source_dim=self.datamodel.dim, dest_dim=self.bin_dim)
            else:
                raise Exception("No such embedding class")

    @abstractmethod
    def get_params(self):
        index_params = super().get_params()
        index_params['num_bits'] = self.num_bits
        index_params['num_tables'] = self.num_tables
        if self.seed is not None:
            index_params['seed'] = self.seed
        if self.needs_embedding:
            index_params['embedding_class'] = self.embedding_class
            if self.embedding_class == 'shp':
                index_params['prrot_iters'] = self.prrot_iters
        return index_params

    def load_from_params(self, params):
        # some hack
        rewrite = False
        if 'id' not in params and 'ind_id' in params:
            params = dict(params)
            params['id'] = params['ind_id']
            rewrite = True
        super_rewrite = super().load_from_params(params)
        rewrite = rewrite or super_rewrite
        # assert self.datamodel.obj_id == params['dm_id']
        self.dataset_indices = params['dataset_indices']

        if self.needs_embedding:
            len_pref = len(self.EMB_PREFIX)
            embedding_keys = {string[len_pref:]: params[string] for string in params.keys()
                              if string[:len_pref] == self.EMB_PREFIX}
            embedding_rewrite = self.embedding.load_from_params(embedding_keys)
            rewrite = rewrite or embedding_rewrite
            if self.seed is not None:
                assert self.embedding.seed == self.seed

        self.index = [{} for i in range(self.num_tables)]
        for table_ind in range(self.num_tables):
            ser_index = params[f'index_{table_ind}']
            for entry in ser_index:
                self.index[table_ind][entry['clus_ind']] = (entry['start_ind'], entry['count'])
        return rewrite

    def get_save_params(self):
        params = super().get_save_params()
        # params['dm_id'] = self.datamodel.obj_id
        params['dataset_indices'] = self.dataset_indices

        for table_ind in range(self.num_tables):
            cur_dict = self.index[table_ind]
            entry_type = np.dtype([('clus_ind', int), ('start_ind', np.uint32), ('count', np.uint32)])
            ser_index = np.zeros(len(cur_dict), dtype=entry_type)

            i = 0
            for clus_ind in cur_dict:
                ser_index[i]['clus_ind'] = clus_ind
                ser_index[i]['start_ind'], ser_index[i]['count'] = cur_dict[clus_ind]
                i += 1

            params[f'index_{table_ind}'] = ser_index
        if self.needs_embedding:
            embedding_params = self.embedding.get_save_params()
            emb_par_w_prefix = {self.EMB_PREFIX + key: embedding_params[key] for key in embedding_params}

            params.update(emb_par_w_prefix)
        return params

    def generate(self):
        super().generate()
        self.index = [{} for i in range(self.num_tables)]
        self.dataset_indices = np.zeros(shape=(self.num_tables, self.datamodel.size_ds), dtype=np.uint32)
        if self.needs_embedding:
            means = None
            if self.datamodel.NEEDS_MEAN_REDUCTION:
                means = np.mean(self.datamodel.dataset, axis=0)
            self.embedding.generate(means=means)
        self._complete_index()

    def _complete_index(self):
        # start_time = time.time()
        while True:
            empty_ind = None
            for i in range(self.num_tables):
                if len(self.index[i]) == 0:
                    empty_ind = i
                    break

            # if no index is empty, we're good.
            if empty_ind is None:
                return

            self._fill_table_index(empty_ind)

            print(f"{Algo.name_from_params(self.get_params())}: indexed table index {empty_ind}")

    def _fill_table_index(self, table_ind):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(f"Filling index {table_ind}, starting time: {current_time}")
        point_to_clusters = np.zeros(self.datamodel.size_ds, dtype=int)
        c_point_to_clusters = np.ctypeslib.as_ctypes(point_to_clusters)
        shared_point_to_clusters = sharedctypes.RawArray(c_point_to_clusters._type_, c_point_to_clusters)

        processes = []
        success_checks = []
        trace = sys.gettrace()
        if trace is None:
            num_processes = mp.cpu_count()
        else:
            num_processes = 1
        # print(mp.cpu_count())
        print(f"creating new index using {num_processes} processes")
        profile = False
        if profile:
            dest_func = self._sp_fill_profile
        else:
            dest_func = self._sp_fill
        dataset = self.datamodel.get_dataset()
        for i in range(num_processes):
            succ_val = mp.Value('i', 0)
            success_checks.append(succ_val)
            proc_slice = slice(i, self.datamodel.size_ds, num_processes)
            process_args = (dataset[proc_slice].copy(),
                            proc_slice,
                            table_ind,
                            shared_point_to_clusters,
                            succ_val)
            processes.append(mp.Process(target=dest_func,
                                        args=process_args))
        for i in range(num_processes):
            processes[i].start()
        for i in range(num_processes):
            processes[i].join()
        for i in range(num_processes):
            assert success_checks[i].value == 1

        point_to_clusters = np.ctypeslib.as_array(shared_point_to_clusters)

        values, counts = np.unique(point_to_clusters, return_counts=True)
        end_indices = np.cumsum(counts)
        start_indices = end_indices - counts
        self.index[table_ind] = {values[i]: (start_indices[i], counts[i]) for i in range(len(values))}

        # the index here is a set of indices into the dataset, sorted by their datawords (with multiplicities)
        sort_inds = np.argsort(point_to_clusters)
        cur_dataset_indices = np.arange(len(point_to_clusters))[sort_inds]
        self.dataset_indices[table_ind, ...] = cur_dataset_indices

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(f"Filled index {table_ind}, end time: {current_time}")

    def _sp_fill_profile(self, *args):
        pr = cProfile.Profile()
        pr.enable()
        pr.runcall(self._sp_fill, *args)
        pr.dump_stats(f'Resources/Profiling/fill_proc_{os.getpid()}')

    def _sp_fill(self, proc_dataset,
                 proc_slice,
                 table_ind,
                 shared_point_to_clusters,
                 succ_val):
        bin_proc_dataset = self.to_bin_space(proc_dataset)
        num_clusters = 1
        cluster_ids = self.get_cluster_ids(bin_proc_dataset, table_ind, num_clusters)

        shared_point_to_clusters_np = np.ctypeslib.as_array(shared_point_to_clusters)
        shared_point_to_clusters_np[proc_slice] = cluster_ids[..., 0]
        succ_val.value = 1

    @abstractmethod
    def get_cluster_ids(self, bin_queries, table_ind, num_clusters):
        pass

    def to_bin_space(self, words):
        """
        Extracts the binary word representation of a dataset word.
        This is either through a binary embedding (real dataset) or simply through indexing (binary dataset).
        """
        if self.needs_embedding:
            return self.embedding.apply(words)
        else:
            assert self.bin_dim <= self.datamodel.dim
            return index_packed_point(words, 0, self.bin_dim)

    @classmethod
    def data_word_to_index(cls, data_word, data_dim):
        """
        A canonical way of converting a byte array to integer for extracting cluster IDs.
        """
        result = np.zeros(shape=data_word.shape[:-1], dtype=int)
        d = data_dim
        for i in range(data_word.shape[-1]):
            curd = min(d, 8)
            assert curd != 0
            result <<= curd
            result += (data_word[..., i] >> (8 - curd))
            d -= curd
        return result


class HashingAlgo(Algo):
    """
    This is a base class for both PCNN and LSH, which embeds the dataset into binary then clusters it.
    """

    INDEX_CLASS = HashingIndex

    def __init__(self, *, num_bits,
                 num_tables,
                 seed=None,
                 embedding_class='hp',
                 **kwargs):
        super().__init__(**kwargs)
        self.bin_dim = kwargs['bin_dim']
        self.num_tables = num_tables
        self.num_bits = num_bits
        self.seed = seed
        self.probe_size = None
        if 'probe_size' in kwargs:
            self.set_probe_size(kwargs['probe_size'])
        self.embedding_class = embedding_class
        if self.embedding_class == 'shp':
            self.prrot_iters = kwargs['prrot_iters']

        # stats
        self.ndis_during_query = 0
        self.num_queries = 0

    def query(self, *, datamodel, query_obj, size_nn):
        """
        handles a set of queries, as given in query_obj.
        :param datamodel: the datamodel to search
        :param query_obj: the query object
        :param size_nn: the number of nearest neighbors to return
        :return:
        """
        self.zero_stats()  # zero the statistics before querying

        queries = query_obj.queries
        num_queries = len(queries)
        processes = []
        trace = sys.gettrace()
        if trace is None:
            num_processes = mp.cpu_count()
        else:
            num_processes = 1
        # num_processes = mp.cpu_count()
        print(num_processes)

        distances = np.zeros(shape=(num_queries, size_nn), dtype=np.float32)
        c_distances = np.ctypeslib.as_ctypes(distances)
        shared_distances = sharedctypes.RawArray(c_distances._type_, c_distances)

        points = np.zeros(shape=(num_queries, size_nn), dtype=int)
        c_points = np.ctypeslib.as_ctypes(points)
        shared_points = sharedctypes.RawArray(c_points._type_, c_points)

        shared_ndis = mp.Value('i', 0)
        success_checks = []
        profile = False
        if profile:
            dest_func = self._sp_query_profile
        else:
            dest_func = self._sp_query

        for i in range(num_processes):
            succ_val = mp.Value('i', 0)
            success_checks.append(succ_val)
            proc_slice = slice(i, num_queries, num_processes)
            # proc_source_queries = None if source_queries is None else source_queries[proc_slice]
            process_args = (queries[proc_slice],
                            datamodel,
                            size_nn,
                            succ_val,
                            shared_distances,
                            shared_points,
                            proc_slice,
                            shared_ndis)
            processes.append(mp.Process(target=dest_func, args=process_args))
        for i in range(num_processes):
            processes[i].start()
        for i in range(num_processes):
            processes[i].join()
        for i in range(num_processes):
            assert success_checks[i].value == 1
        distances = np.ctypeslib.as_array(shared_distances)
        points = np.ctypeslib.as_array(shared_points)

        # update stats
        self.ndis_during_query += shared_ndis.value
        self.num_queries += len(queries)

        return distances, points

    def _sp_query_profile(self, *args):
        pr = cProfile.Profile()
        pr.enable()
        pr.runcall(self._sp_query, *args)
        pr.dump_stats(f'Resources/Profiling/query_proc_{os.getpid()}')

    def _sp_query(self, queries,
                  datamodel,
                  size_nn,
                  succ_val,
                  shared_distances=None,
                  shared_points=None,
                  proc_slice=None,
                  shared_ndis=None,
                  ):
        """
        this function is run concurrently with other instances of this function, and handles a set of queries.
        It finds the probe_size clusters to probe for each query, then compares the distance of the query to the points in these clusters, and chooses the closest
        size_nn points. These get written to the shared buffers shared_points and shared_distances.
        :param queries: the queries
        :param datamodel:
        :param size_nn:
        :param succ_val:
        :param shared_distances:
        :param shared_points:
        :param proc_slice:
        :param shared_ndis:
        :return:
        """
        start_ndis = self.ndis_during_query
        num_queries = len(queries)

        probe_per_table = self.probe_size // self.num_tables
        cluster_ids = np.zeros(shape=(num_queries, self.num_tables, probe_per_table), dtype=int)
        bin_queries = self.index_obj.to_bin_space(queries)
        for num_table in range(self.num_tables):
            cluster_ids[:, num_table, :] = self.index_obj.get_cluster_ids(bin_queries, num_table,
                                                                          num_clusters=probe_per_table)

        distance_results = np.zeros((num_queries, size_nn))
        point_results = np.zeros_like(distance_results, dtype=int)
        for i in range(num_queries):
            distance_results[i, :], point_results[i, :] = self._query_single(queries[i], datamodel, size_nn,
                                                                             cluster_ids[i])
        result = distance_results, point_results
        if shared_distances is not None:
            shared_dist_np = np.ctypeslib.as_array(shared_distances)
            shared_dist_np[proc_slice] = distance_results
            shared_points_np = np.ctypeslib.as_array(shared_points)
            shared_points_np[proc_slice] = point_results

            # update ndis
            end_ndis = self.ndis_during_query
            ndis_diff = end_ndis - start_ndis
            shared_ndis.value += ndis_diff
        succ_val.value = 1
        return result

    def _query_single(self, query, datamodel, size_nn, cluster_ids):
        """
        Given a query and the chosen cluster IDs to probe for that query, returns the closest size_nn points to the query.
        This happens by comparing the query with the points in the cluster (using the original metric).
        """
        num_inds_with_dups = 0
        for table_ind in range(cluster_ids.shape[0]):
            for query_clus_id_ind in range(cluster_ids.shape[1]):
                clus_id = cluster_ids[table_ind, query_clus_id_ind]
                if clus_id in self.index_obj.index[table_ind]:
                    start_ind, count = self.index_obj.index[table_ind][clus_id]
                    end_ind = start_ind + count
                    num_inds_with_dups += end_ind - start_ind

        point_indices_with_dups = np.zeros((num_inds_with_dups,), dtype=int)

        # add points to test
        point_ind = 0
        for table_ind in range(cluster_ids.shape[0]):
            for query_clus_id_ind in range(cluster_ids.shape[1]):
                clus_id = cluster_ids[table_ind, query_clus_id_ind]

                if clus_id in self.index_obj.index[table_ind]:
                    start_ind, count = self.index_obj.index[table_ind][clus_id]
                    end_ind = start_ind + count
                    clus_size = end_ind - start_ind
                    point_indices_with_dups[point_ind: point_ind + clus_size] = self.index_obj.dataset_indices[
                                                                                table_ind, start_ind: end_ind]
                    point_ind += clus_size

        # remove duplicates
        point_indices = np.unique(point_indices_with_dups)

        # get points and calculate distances
        points = datamodel.dataset[point_indices, :]
        if datamodel.METRIC == BaseDataModel.METRIC_HAMMING:
            distances = get_binary_dist(query[np.newaxis, ...], points)
        elif datamodel.METRIC == BaseDataModel.METRIC_L2:
            distances = get_l2_dist(query[np.newaxis, ...], points)
        elif datamodel.METRIC == BaseDataModel.METRIC_IP:
            distances = get_cosine_dist(query[np.newaxis, ...], points)
        else:
            raise Exception('HashingAlgo: real_datamodel data_type unimplemented')

        self.ndis_during_query += len(point_indices)

        return self._choose_best_results(distances=distances, point_indices=point_indices, size_nn=size_nn)

    @classmethod
    def _choose_best_results(cls, distances, point_indices, size_nn):
        # choose the best results.
        if len(distances) < size_nn:
            result_distances = distances
            result_indices = point_indices
        else:
            part_inds = np.argpartition(distances, size_nn - 1)
            result_distances = distances[part_inds][:size_nn]
            result_indices = point_indices[part_inds][:size_nn]

        # sort results
        sort_inds = np.argsort(result_distances, axis=-1)
        result_distances = np.take_along_axis(result_distances, sort_inds, axis=-1)
        result_indices = np.take_along_axis(result_indices, sort_inds, axis=-1)

        # pad the number of results if there aren't size_nn of them.
        if len(result_distances) < size_nn:
            result_distances = np.pad(result_distances, [(0, size_nn - len(result_distances))],
                                      constant_values=[(0, 0x7fffffff)])
            result_indices = np.pad(result_indices, [(0, size_nn - len(result_indices))], constant_values=[(0, -1)])

        return result_distances, result_indices

    def get_index_config(self):
        config = {
            'num_bits': self.num_bits,
            'num_tables': self.num_tables,
            'seed': self.seed,
        }
        config['embedding_class'] = self.embedding_class
        if self.embedding_class == 'shp':
            config['prrot_iters'] = self.prrot_iters
        return config

    """ Stat Methods """

    def get_ndis(self):
        return self.ndis_during_query

    def get_num_queries(self):
        return self.num_queries

    def get_nop(self):
        return self.ndis_during_query + self.probe_size * self.num_queries

    def zero_stats(self):
        self.ndis_during_query = 0
        self.num_queries = 0

    def set_index(self, index_obj):
        self.index_obj = index_obj

    def get_index_params(self):
        return self.index_obj.get_params()

    """ Param Getters"""

    def get_probing_params(self):
        probing_params = super().get_probing_params()
        probing_params['probe_size'] = self.probe_size
        return probing_params

    """ Abstract Methods """

    @abstractmethod
    def set_probe_size(self, probe_size):
        pass
