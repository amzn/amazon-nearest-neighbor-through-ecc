from abc import ABC, abstractmethod
import random
import os
import numpy as np


class SavedObject(ABC):
    """
    This abstract class represents an object whose content is cached in a file.
    """

    def __init__(self, **kwargs):
        """
        The initialization creates an empty object with the necessary parameters to either generate or load content.
        """
        self.obj_id = None

    ############### generation ###############
    @abstractmethod
    def generate(self):
        """
        generates new content given the parameters of the object.
        For this base class, this only generates a (probabilistically) unique ID for that content; overriding methods in child classes will call this method, and perform additional actions.
        """
        self.obj_id = self._gen_id()

    def _gen_id(self):
        """
        Returns a new random obj_id.
        :return:
        """
        return random.randrange(2 ** 128)

    ############### saving ###############
    def save(self, filepath):
        assert os.path.splitext(filepath)[1] == '.npz'
        params = self.get_save_params()
        np.savez(filepath, **params)

    @abstractmethod
    def get_save_params(self) -> dict:
        """
        returns a dictionary mapping from strings (content names) to arraylike content (ndarray, int, float...).
        This dictionary is the content to be saved.
        :return:
        """
        return {'id': self.obj_id}

    ############### loading ###############

    @abstractmethod
    def load_from_params(self, params):
        """
        Loads the object from the dictionary 'params'. Returns 'true' if the saved copy should be updated after loading
        (useful for adding missing fields).
        :param params:
        :return:
        """
        rewrite = False
        if 'id' not in params:
            self.obj_id = self._gen_id()
            rewrite = True
        else:
            self.obj_id = params['id']

        return rewrite

    def load(self, filepath, do_save=True):
        """
        Loads the object from file, and possibly rewrites it if fields need to be added.
        """

        file_cont = np.load(filepath, allow_pickle=True)
        rewrite = self.load_from_params(file_cont)
        if rewrite and do_save:
            self.save(filepath)

    ############### obtaining ###############
    def obtain(self, filepath, do_save=True):
        """
        loads the object from the given path if exists, otherwise generates the object (possibly saving it to file).
        """
        if os.path.exists(filepath):
            self.load(filepath, do_save)
        else:
            self.generate()
            if do_save:
                self.save(filepath)
