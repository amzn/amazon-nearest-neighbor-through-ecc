from .real_dm import RealDataModel
import numpy as np
import random


class GloveDataModel(RealDataModel):
    DATASET_FILENAME = r'glove.twitter.27B.200d.txt'
    SOURCE_DIM = 200
    GLOVE = 'Glove'

    SOURCE_NAME = GLOVE
    SUB_MEAN = False

    # the maximum number of query_dict that the real_datamodel can generate.
    MAX_QUERIES = 10000

    # some info
    VOC_SIZE = 1193514
    DS_SIZE = VOC_SIZE - MAX_QUERIES

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.query_indices = None

    def read_source_words(self, *, source_filename, is_query, num_to_read=None):
        # source_filename = cls.SOURCE_DATASET_FILES[source_name]
        source_length = self._get_source_len(source_filename)
        if self.query_indices is None:
            self.query_indices = set(random.sample(range(source_length), self.MAX_QUERIES))

        if is_query:
            max_size = self.MAX_QUERIES
        else:
            max_size = source_length - self.MAX_QUERIES

        # if number of points to read is not specified, read everything.
        if num_to_read is None:
            num_to_read = max_size

        output = np.zeros(shape=(num_to_read, self.SOURCE_DIM), dtype=np.float32)
        with open(self.SOURCE_PATH + source_filename, 'r') as f:
            num_read = 0
            for line_num in range(source_length):
                if num_read == num_to_read:
                    break
                line = f.readline()
                if is_query and line_num not in self.query_indices:
                    continue
                if not is_query and line_num in self.query_indices:
                    continue
                # a line should contain the word name and the floating-point vector entries
                entry_string = line.split(' ')[1:]
                assert len(entry_string) == self.SOURCE_DIM
                vector = np.array([float(s) for s in entry_string], dtype=np.float32)
                output[num_read] = vector
                num_read += 1
        return output

    def _get_source_len(self, source_filename):
        length = 0
        with open(self.SOURCE_PATH + source_filename, 'r') as f:
            for line in f:
                length += 1
        return length

    def read_source_dataset(self, *, source_name, num_to_read=None):
        return self.read_source_words(source_filename=self.DATASET_FILENAME, num_to_read=num_to_read, is_query=False)

    def read_source_queries(self, *, source_name, num_to_read=None):
        return self.read_source_words(source_filename=self.DATASET_FILENAME, num_to_read=num_to_read, is_query=True)

    def save_datamodel(self, filename):
        filepath = self.DATAMODEL_PATH + filename
        np.savez(filepath,
                 dm_id=self.dm_id,
                 transform=self.binary_transform,
                 dataset=self.dataset,
                 query_indices=np.array(list(self.query_indices), dtype=int),
                 source_dataset=self.source_dataset)

    def load_datamodel_from_dict(self, file_cont):
        rewrite = super().load_datamodel_from_dict(file_cont)
        self.query_indices = set(file_cont['query_indices'])
        return rewrite
