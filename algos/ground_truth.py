import os
import numpy as np
from . import ExactSearch
from .saved_object import SavedObject


class GroundTruth(SavedObject):

    def __init__(self, datamodel, query_obj, size_nn):
        super().__init__()
        self.datamodel = datamodel
        self.query_obj = query_obj
        self.size_nn = size_nn

    def generate(self):
        super().generate()
        brute = ExactSearch(dim=self.datamodel.dim)
        brute.create_index(datamodel=self.datamodel)
        distances, points = brute.query(datamodel=self.datamodel, query_obj=self.query_obj, size_nn=self.size_nn)
        self.distances = distances
        self.points = points

    def get_save_params(self):
        params = super().get_save_params()
        params['dm_id'] = self.datamodel.obj_id
        params['query_id'] = self.query_obj.obj_id
        params['distances'] = self.distances
        params['points'] = self.points
        return params

    def load_from_params(self, params):
        rewrite = super().load_from_params(params)
        assert params['dm_id'] == self.datamodel.obj_id
        assert params['query_id'] == self.query_obj.obj_id
        self.distances = params['distances']
        self.points = params['points']
        return rewrite
