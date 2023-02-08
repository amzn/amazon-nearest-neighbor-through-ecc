from .real_dm import RealDataModel, RealQuery, FileReader
from .base_dm import BaseDataModel
import numpy as np


class YandexDeepQuery(RealQuery):
    METRIC = BaseDataModel.METRIC_L2
    SOURCE_QUERY_FILE = r'YandexDeep_query_10K.fbin'

    def read_source_queries(self, *, num_to_read):
        source_filename = self.SOURCE_QUERY_FILE
        source_filepath = self.SOURCE_PATH + source_filename
        return FileReader.read_source_words(source_filepath=source_filepath, num_to_read=num_to_read)


class YandexDeepDataModel(RealDataModel):
    # some source dataset names

    METRIC = BaseDataModel.METRIC_L2
    YANDEX_DEEP_DIM = 96
    SOURCE_TYPE = np.float32
    SOURCE_DATASET_FILE = r'YandexDeep_learn_350M.fbin'
    NEEDS_MEAN_REDUCTION = False
    source_name = RealDataModel.YANDEX_DEEP
    QUERY_CLASS = YandexDeepQuery

    # def read_source_words(self, *, source_filename, num_to_read):
    #     with open(self.SOURCE_PATH + source_filename, 'rb') as f:
    #         num_points = np.fromfile(f, dtype=np.uint32, count=1)
    #         assert num_to_read <= num_points
    #         source_dimension = int(np.fromfile(f, dtype=np.uint32, count=1))
    #         data_type = self.SOURCE_TYPE
    #         data = np.fromfile(f, dtype=data_type, count=num_to_read * source_dimension)
    #         data = data.reshape(num_to_read, source_dimension).astype(np.float32)
    #     return data

    def read_source_dataset(self, *, num_to_read):
        source_filename = self.SOURCE_DATASET_FILE
        source_filepath = self.SOURCE_PATH + source_filename
        return FileReader.read_source_words(source_filepath=source_filepath, num_to_read=num_to_read)
