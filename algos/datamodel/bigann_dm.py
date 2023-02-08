from .real_dm import RealDataModel, RealQuery, FileReader
from .base_dm import BaseDataModel
import numpy as np


class BIGANNQuery(RealQuery):
    METRIC = BaseDataModel.METRIC_L2
    SOURCE_QUERY_FILE = r'BIGANN_query_10K.u8bin'

    def read_source_queries(self, *, num_to_read):
        source_filename = self.SOURCE_QUERY_FILE
        source_filepath = self.SOURCE_PATH + source_filename
        return FileReader.read_source_words(source_filepath=source_filepath, num_to_read=num_to_read)


class BIGANNDataModel(RealDataModel):
    # some source dataset names

    METRIC = BaseDataModel.METRIC_L2
    BIGANN_DIM = 128
    NEEDS_MEAN_REDUCTION = True
    SOURCE_TYPE = np.uint8
    SOURCE_DATASET_FILE = r'BIGANN_learn_100M.u8bin'
    SOURCE_QUERY_FILE = r'BIGANN_query_10K.u8bin'
    source_name = RealDataModel.BIGANN
    QUERY_CLASS = BIGANNQuery

    def read_source_dataset(self, *, num_to_read):
        source_filename = self.SOURCE_DATASET_FILE
        source_filepath = self.SOURCE_PATH + source_filename
        return FileReader.read_source_words(source_filepath=source_filepath, num_to_read=num_to_read)
