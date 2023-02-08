import random
from .base_dm import BaseDataModel
from .bigann_dm import BIGANNDataModel
from src.algos.utils.standard_utils import *
import numpy as np
from .base_dm import Query


class BinarizedQuery(Query):

    def generate(self):
        super().generate()
        real_dm = self.datamodel.real_dm
        query_obj = real_dm.QUERY_CLASS(datamodel=real_dm, num_queries=self.num_queries)
        query_obj.generate()
        self.queries = self.datamodel.binary_embedding.apply(query_obj.queries)


class BinarizedDataModel(BaseDataModel):
    METRIC = BaseDataModel.METRIC_HAMMING
    QUERY_CLASS = BinarizedQuery

    EMB_PREFIX = 'hpemb_'

    def __init__(self, *, real_dm, seed=None, **kwargs):
        kwargs['size_ds'] = real_dm.size_ds
        super().__init__(**kwargs)
        self.real_dm = real_dm
        self.seed = seed
        self.binary_embedding = HyperplaneEmbedding(source_dim=self.real_dm.dim,
                                                    dest_dim=self.dim,
                                                    seed=seed)

    def generate(self):
        super().generate()
        if self.real_dm.NEEDS_MEAN_REDUCTION:
            means = np.mean(self.real_dm.dataset, axis=0)
        else:
            means = None
        self.binary_embedding.generate(means=means)
        self.dataset = self.binary_embedding.apply(self.real_dm.dataset)

    def get_save_params(self):
        save_params = super().get_save_params()
        save_params['real_dm_id'] = self.real_dm.obj_id

        # add embedding save params
        embedding_params = self.binary_embedding.get_save_params()
        emb_par_w_prefix = {self.EMB_PREFIX + key: embedding_params[key] for key in embedding_params}
        save_params.update(emb_par_w_prefix)
        return save_params

    def load_from_params(self, params):
        rewrite = super().load_from_params(params)
        assert self.real_dm.obj_id == params['real_dm_id']

        # load embedding
        len_pref = len(self.EMB_PREFIX)
        embedding_keys = {string[len_pref:]: params[string] for string in params.keys()
                          if string[:len_pref] == self.EMB_PREFIX}
        embedding_rewrite = self.binary_embedding.load_from_params(embedding_keys)
        rewrite = rewrite or embedding_rewrite
        if self.seed is not None:
            assert self.binary_embedding.seed == self.seed
        return rewrite

    def get_params(self):

        params = {
            'name': 'binarized_dataset',
            'bindim': self.dim,
            'seed': self.binary_embedding.seed
        }
        real_params = self.real_dm.get_params()
        params['real_name'] = real_params['name']
        del real_params['name']
        params.update(real_params)
        return params
