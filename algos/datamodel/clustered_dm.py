from src.algos.utils.standard_utils import generate_binary_points, generate_biased_binary_points
from .base_dm import BaseDataModel
import math
import numpy as np


class ClusteredDataModel(BaseDataModel):

    def __init__(self, **kwargs):
        super().__init__(dim=kwargs['dim'], size_ds=kwargs['size_ds'])
        self.leaders = None
        self.noise_exp = kwargs['noise_exp']
        self.ppc = kwargs['ppc']

        self.noise_ratio = self.noise_exp / self.dim

    @classmethod
    def _get_model_filename(cls, **kwargs):
        size_ds = kwargs['size_ds']
        dim = kwargs['dim']
        ppc = kwargs['ppc']
        noise_exp = kwargs['noise_exp']
        return f'clustered_dataset_{size_ds}_{dim}_ppc{ppc}_nexp{noise_exp}.npz'

    def get_filename(self):
        return self._get_model_filename(size_ds=self.size_ds, dim=self.dim, ppc=self.ppc, noise_exp=self.noise_exp)

    @classmethod
    def _generate_datamodel(cls, **kwargs):
        dim = kwargs['dim']
        ppc = kwargs['ppc']
        size_ds = kwargs['size_ds']
        noise_exp = kwargs['noise_exp']
        datamodel = cls(dim=dim, ppc=ppc, size_ds=size_ds, noise_exp=noise_exp)
        noise_ratio = noise_exp / dim
        num_leaders = math.ceil(size_ds / ppc)
        datamodel.leaders = generate_binary_points(num_points=num_leaders, dim=dim)

        packed_dim = math.ceil(dim / 8)
        result = generate_biased_binary_points(num_points=size_ds, dim=dim, prob1=noise_ratio)

        cur_ind = 0
        for i, lead in enumerate(datamodel.leaders):
            start_ind = ppc * i
            end_ind = min(start_ind + ppc, size_ds)
            result[start_ind:end_ind] ^= lead
        datamodel.dataset = result

        return datamodel

    def save_datamodel(self, filename):
        filepath = self.DATAMODEL_PATH + filename
        np.savez(filepath, leaders=self.leaders, dataset=self.dataset)

    @classmethod
    def load_datamodel(cls, filename, **kwargs):
        '''
        :key dim
        :key size_ds
        :key ppc
        :key noise_exp
        :return: the real_datamodel object
        '''
        datamodel = cls(size_ds=kwargs['size_ds'], dim=kwargs['dim'], noise_exp=kwargs['noise_exp'], ppc=kwargs['ppc'])
        filepath = cls.DATAMODEL_PATH + filename
        file_cont = np.load(filepath)
        datamodel.leaders, datamodel.dataset = file_cont['leaders'], file_cont['dataset']
        return datamodel

    def generate_queries(self, **kwargs):
        num_queries = kwargs['num_queries']

        queries = generate_biased_binary_points(num_points=num_queries, dim=self.dim, prob1=self.noise_ratio)
        rng = np.random.default_rng()
        chosen_leaders = rng.choice(self.leaders, num_queries)
        queries ^= chosen_leaders
        query_dict = {'queries': queries}
        return query_dict

    def save_queries(self, query_dict):
        num_queries = len(query_dict['queries'])
        querypath = self._get_query_filepath(num_queries=num_queries)
        with open(querypath, 'wb') as f:
            np.savez(f, queries=query_dict['queries'])
