import os

from .code_mask import BSCCodeMask
from .ground_truth import GroundTruth


class DataModelFactory:
    """
    This class:
    * generates datamodels.
    * saves them to file according to a file naming scheme.
    * returns them to the caller.
    """
    DATAMODEL_FILEPATH = 'Resources/Datasets/'

    @classmethod
    def get_datamodel(cls, dm_class, **dm_ctor_args):
        datamodel = dm_class(**dm_ctor_args)
        filename = cls.get_dm_filename(datamodel)
        filepath = cls.DATAMODEL_FILEPATH + filename
        datamodel.obtain(filepath)
        return datamodel, filename

    @classmethod
    def get_dm_filename(cls, datamodel):
        dm_params = datamodel.get_params()
        return cls.get_dm_filename_from_params(dm_params)

    @classmethod
    def get_dm_filename_from_params(cls, dm_params):
        string_parts = [str(value) for value in dm_params.values()]
        filename = '_'.join(string_parts) + '.npz'
        return filename


class IndexFactory:
    INDEX_FILEPATH = 'Resources/Indices/'

    @classmethod
    def get_index(cls, index_class, ephemeral=False, **kwargs):
        index = index_class(**kwargs)
        filepath = cls.get_index_filepath(index)
        index.obtain(filepath, do_save=(not ephemeral))
        return index

    @classmethod
    def get_index_filename(cls, index):
        dm_filename = DataModelFactory.get_dm_filename(index.datamodel)
        dm_filename_noext = os.path.splitext(dm_filename)[0]
        index_params = index.get_params()
        name = index_params['algo_name']
        del index_params['algo_name']
        # default is hyperplanes
        if 'embedding_class' in index_params and index_params['embedding_class'] == 'hp':
            del index_params['embedding_class']
        param_strings = [f'{key}={value}' for key, value in index_params.items()]
        ext = '.npz'
        return f'{name}_index: ' + '|'.join(param_strings) + f' ({dm_filename_noext})' + ext

    @classmethod
    def get_index_filepath(cls, index):
        filename = cls.get_index_filename(index)
        return cls.INDEX_FILEPATH + filename


class QueryFactory:
    QUERY_FILEPATH = 'Resources/Queries/'

    @classmethod
    def get_query(cls, query_class, dm_filename, **kwargs):
        query = query_class(**kwargs)
        filename = cls.get_query_filename(query, dm_filename)
        filepath = cls.QUERY_FILEPATH + filename
        query.obtain(filepath)
        return query, filename

    @classmethod
    def get_query_filename(cls, query, dm_filename):
        dm_filename_noext = os.path.splitext(dm_filename)[0]
        num_queries = query.num_queries
        filename = f'queries_nq{num_queries} ({dm_filename_noext}).npz'
        return filename


class GroundTruthFactory:
    GROUND_FILEPATH = 'Resources/Ground/'

    @classmethod
    def get_gt(cls, *, datamodel, query_obj, query_filename, size_nn):
        gt = GroundTruth(datamodel, query_obj, size_nn)
        filename = cls.get_gt_filename(gt, query_filename)
        filepath = cls.GROUND_FILEPATH + filename
        gt.obtain(filepath)
        return gt, filename

    @classmethod
    def get_gt_filename(cls, gt, query_filename):
        query_filename_noext = os.path.splitext(query_filename)[0]
        filename = f"ground_sizenn{gt.size_nn} ({query_filename_noext}).npz"
        return filename


class MaskFactory:
    MASK_FILEPATH = 'Resources/Codes/Masks/'

    @classmethod
    def get_mask(cls, *, N, K):
        mask_obj = BSCCodeMask(N, K)
        filename = cls.get_mask_filename(mask_obj)
        filepath = cls.MASK_FILEPATH + filename
        mask_obj.obtain(filepath)
        return mask_obj, filename

    @classmethod
    def get_mask_filename(cls, mask_obj):
        filename = f'bsc_mask_{mask_obj.N}_{mask_obj.K}.npz'
        return filename
