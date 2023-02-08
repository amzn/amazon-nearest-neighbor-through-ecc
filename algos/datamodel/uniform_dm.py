from .base_dm import BaseDataModel, Query
from src.algos.utils.standard_utils import generate_binary_points


class UniformQuery(Query):
    SOURCE_PATH = 'Resources/Sources/'

    def generate(self):
        super().generate()
        self.queries = generate_binary_points(num_points=self.num_queries, dim=self.datamodel.dim)


class UniformDataModel(BaseDataModel):
    METRIC = BaseDataModel.METRIC_HAMMING
    NEEDS_MEAN_REDUCTION = False
    QUERY_CLASS = UniformQuery

    def get_params(self):
        filename_params = super().get_params()
        filename_params['name'] = 'uniform_dataset'
        filename_params['size_ds'] = self.size_ds
        filename_params['dim'] = self.dim
        return filename_params

    def generate(self):
        super().generate()
        self.dataset = generate_binary_points(num_points=self.size_ds, dim=self.dim)

    def generate_queries(self, *, num_queries):
        return generate_binary_points(num_points=num_queries, dim=self.dim)
