from .real_dm import RealDataModel, RealQuery, FileReader
from .base_dm import BaseDataModel
import numpy as np


class YandexTTIQuery(RealQuery):
    SOURCE_QUERY_FILE = r'YandexTTI_query_100K.fbin'
    METRIC = BaseDataModel.METRIC_IP

    def read_source_queries(self, *, num_to_read):
        source_filename = self.SOURCE_QUERY_FILE
        source_filepath = self.SOURCE_PATH + source_filename
        return FileReader.read_source_words(source_filepath=source_filepath, num_to_read=num_to_read)


class YandexTTIDataModel(RealDataModel):
    # some source dataset names
    METRIC = BaseDataModel.METRIC_IP
    YANDEX_TTI_DIM = 200
    SOURCE_TYPE = np.float32
    NEEDS_MEAN_REDUCTION = False
    QUERY_CLASS = YandexTTIQuery

    SOURCE_DATASET_FILE = r'YandexTTI_learn_50M.fbin'

    source_name = RealDataModel.YANDEX_TTI

    def read_source_dataset(self, *, num_to_read):
        source_filename = self.SOURCE_DATASET_FILE
        source_filepath = self.SOURCE_PATH + source_filename
        return FileReader.read_source_words(source_filepath=source_filepath, num_to_read=num_to_read)
