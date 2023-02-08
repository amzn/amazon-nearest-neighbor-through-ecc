from abc import abstractmethod
from .base_dm import BaseDataModel, Query
import numpy as np
import os


class RealDataModel(BaseDataModel):
    SOURCE_PATH = 'Resources/Sources/'
    source_name = None
    NEEDS_MEAN_REDUCTION = None

    # some source names
    BIGANN = 'BIGANN'
    GLOVE = 'Glove'
    YANDEX_TTI = 'YandexTTI'
    YANDEX_DEEP = 'YandexDeep'

    def __init__(self, *, size_ds, dim):
        super().__init__(size_ds=size_ds, dim=dim)

    ######################## Source Reading Abstract Methods ###############################
    @abstractmethod
    def read_source_dataset(self, *, num_to_read):
        pass

    ##################### filename functions ##########################
    def get_params(self):
        filename_params = super().get_params()
        filename_params['name'] = 'file_dataset'
        filename_params['source_name'] = self.source_name
        filename_params['size_ds'] = self.size_ds
        filename_params['dim'] = self.dim
        return filename_params

    ##################### SavedObject functions ##########################

    def generate(self):
        super().generate()
        source_words = self.read_source_dataset(num_to_read=self.size_ds)
        self.dataset = source_words.astype(np.float32)
        assert self.dataset.shape[-1] == self.dim


class RealQuery(Query):
    SOURCE_PATH = 'Resources/Sources/'

    def generate(self):
        super().generate()
        self.queries = self.read_source_queries(num_to_read=self.num_queries)


class FileReader:

    @classmethod
    def read_source_words(cls, *, source_filepath, num_to_read):
        ext = os.path.splitext(source_filepath)[1]
        if ext == '.fbin':
            data_type = np.float32
        elif ext == '.u8bin':
            data_type = np.uint8
        else:
            raise Exception('wrong format')
        with open(source_filepath, 'rb') as f:
            num_points = np.fromfile(f, dtype=np.uint32, count=1)
            assert num_to_read <= num_points
            source_dimension = int(np.fromfile(f, dtype=np.uint32, count=1))
            data = np.fromfile(f, dtype=data_type, count=num_to_read * source_dimension)
            data = data.reshape(num_to_read, source_dimension).astype(np.float32)
        return data
