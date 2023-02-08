import os
from abc import ABC, abstractmethod

import numba
import numpy as np
import math
import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl
from src.algos.saved_object import SavedObject

UNIMP = r"""raise Exception(f'{self.__class__}: function {inspect.currentframe().f_code.co_name} is unimplemented')"""


def get_binary_dist(p1, p2):
    """
    Returns the distance between two points of the same dimension, simply by xoring and counting 1's.
    """
    p3 = np.bitwise_xor(p1, p2)
    result = np.zeros(shape=p3.shape[:-1], dtype=int)
    for j in range(8):
        result += np.count_nonzero(p3 & 1, axis=-1)
        p3 >>= 1

    return result


def get_l2_dist(p1, p2):
    return np.linalg.norm(p2 - p1, axis=-1)


def get_cosine_dist(p1, p2):
    return 1 - (p1 @ p2.T).reshape(-1)


def generate_binary_points(*, num_points, dim) -> np.ndarray:
    """
    This function generates random binary points.
    The points are represented in a bit-packed ndarray of type np.uint8.
    The arguments are:
    :key num_points: the number of points in the dataset.
    :key dim: the dimension of the dataset. Assumption: this is a multiple of 8.
    """

    rng = np.random.default_rng()
    packed_dim = math.ceil(dim / 8)

    array = rng.integers(0, 256, size=(num_points, packed_dim), dtype=np.uint8)

    # zero out padding bits
    bad_bits = packed_dim * 8 - dim
    mask = '1' * (8 - bad_bits) + '0' * bad_bits
    mask = int(mask, 2)
    array[:, -1] &= mask

    return array


def generate_biased_binary_points(*, num_points, dim, prob1) -> np.ndarray:
    """
    This function generates random binary points.
    The points are represented in a bit-packed ndarray of type np.uint8.
    The arguments are:
    :key num_points: the number of points in the dataset.
    :key dim: the dimension of the dataset.
    :key prob1: the probability that each bit equals one
    """
    packed_dim = math.ceil(dim / 8)
    array = np.zeros(shape=(num_points, packed_dim), dtype=np.uint8)
    rng = np.random.default_rng()

    for j in range(8):
        random_floats = rng.random(size=(num_points, packed_dim))
        random_bits = (random_floats <= prob1).astype(np.uint8)
        array <<= 1
        array ^= random_bits

    # zero out the unnecessary bits
    bad_bits = packed_dim * 8 - dim
    mask = '1' * (8 - bad_bits) + '0' * bad_bits
    mask = int(mask, 2)
    array[:, -1] &= mask

    return array


def pack_point(point, dim):
    return np.packbits(point, axis=-1)


def unpack_point(point, dim):
    unpacked = np.unpackbits(point, axis=-1, count=dim)
    return unpacked


def index_packed_point(point, start, end):
    start_byte_ind = start // 8
    start_offset = start % 8
    end_offset = start_offset + (end - start)
    end_byte_ind = (end + 7) // 8
    ex_words = point[..., start_byte_ind:end_byte_ind]
    unpacked = unpack_point(ex_words, dim=end_offset)[..., start_offset:]
    return pack_point(unpacked, dim=end - start)


class BinaryEmbedding(SavedObject):
    def __init__(self, source_dim, dest_dim, **kwargs):
        self.source_dim = source_dim
        self.dest_dim = dest_dim

    @abstractmethod
    def apply(self, words):
        pass


class AutoencoderEmbedding(BinaryEmbedding):
    """
    This class implements an autoencoder-based binary embedding, as shown in
    "Near-lossless binarization of word embeddings" by Tissier et al. (AAAI 2019)
    """
    MODELS_PATH = 'Resources/Models/'

    class LitAutoEncoder(pl.LightningModule):
        def __init__(self, start_dim, bin_dim):
            super().__init__()
            self.bin_dim = bin_dim
            self.start_dim = start_dim
            self.dec_layer = nn.Linear(bin_dim, start_dim, bias=True)
            self.ind_view = self.dec_layer.weight.detach().requires_grad_(False)
            self.decoder = nn.Sequential(self.dec_layer, nn.Tanh())

        def encode(self, x):
            values = torch.Tensor([0])
            return torch.heaviside(F.linear(x, self.ind_view.t()), values=values)

        def forward(self, b_vals):
            x_hat = self.decoder(b_vals)
            return x_hat

        def training_step(self, batch, batch_idx):
            return self._common_step(batch, batch_idx, "train")

        def validation_step(self, batch, batch_idx):
            self._common_step(batch, batch_idx, "val")

        def test_step(self, batch, batch_idx):
            self._common_step(batch, batch_idx, "test")

        def predict_step(self, batch, batch_idx, dataloader_idx=None):
            x = self._prepare_batch(batch)
            b_vals = self.encode(x)
            return self(b_vals)

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
            return optimizer

        def _prepare_batch(self, batch):
            x = batch
            #         print(batch)
            return x.view(x.size(0), -1)

        def _common_step(self, batch, batch_idx, stage: str):
            x = self._prepare_batch(batch)
            b_vals = self.encode(x)
            b_vals = torch.Tensor(b_vals)
            #         print(type(b_vals))
            batch_size = batch.shape[0]
            rec_loss = F.mse_loss(x, self(b_vals), reduction='sum')
            reg_loss = F.mse_loss(torch.matmul(self.dec_layer.weight, self.dec_layer.weight.t()),
                                  torch.eye(self.start_dim), reduction='sum') / 2
            loss = rec_loss + reg_loss
            self.log(f"{stage}_loss", loss, on_step=True)
            return loss

    def __init__(self, size_ds, epochs=3, seed=42, **kwargs):
        super().__init__(**kwargs)
        assert (seed == 42)
        self.seed = seed
        self.epochs = epochs
        self.size_ds = size_ds
        model_path = f'{self.MODELS_PATH}YandexDeepStrongerRegularizedEmbedding_sizeds={self.size_ds}_bindim={self.dest_dim}_epochs={self.epochs}'
        models_in_path = [filename for filename in os.listdir(model_path) if filename.endswith('.ckpt')]
        if len(models_in_path) != 1:
            raise Exception('number of models in path is wrong')

        self.model_path = f'{model_path}/{models_in_path[0]}'
        self.model = self.LitAutoEncoder.load_from_checkpoint(self.model_path, start_dim=self.source_dim,
                                                              bin_dim=self.dest_dim)
        print(f'StrongerRegularizedAutoencoderEmbedding loaded model from {model_path}')

    def apply(self, words):
        unpacked_result = self.model.encode(torch.Tensor(words.copy())).numpy().astype(np.int8)
        return pack_point(unpacked_result, self.dest_dim)

    def generate(self, means=None):
        super().generate()

    def get_save_params(self):
        """
        returns a dictionary mapping from strings (content names) to arraylike content (ndarray, int, float...).
        This dictionary is the content to be saved.
        :return:
        """
        save_params = super().get_save_params()
        save_params['seed'] = self.seed
        return SavedObject.get_save_params(self)

    def load_from_params(self, params):
        rewrite = super().load_from_params(params)
        if 'seed' not in params or params['seed'] == True:
            self.seed = 42
            rewrite = True
        return rewrite


class HyperplaneEmbedding(BinaryEmbedding):
    """
    This class implements the hyperplane embedding of
    "Similarity Estimation Techniques from Rounding Algorithms" by Charikar (STOC 2002)
    """

    def __init__(self, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.means = None
        self.hyperplane_matrix = None
        self.seed = seed

    def apply(self, words):
        normed_source = words.copy()
        if self.means is not None:
            normed_source -= self.means

        hyper_res = normed_source @ self.hyperplane_matrix.T
        return pack_point((hyper_res < 0).astype(np.int8), self.dest_dim)

    ############### saved object things ###############

    def generate(self, means=None):
        SavedObject.generate(self)
        if means is not None:
            assert means.shape == (self.source_dim,)
            self.means = means
        rng = np.random.default_rng(seed=self.seed)
        self.hyperplane_matrix = rng.standard_normal(size=(self.dest_dim, self.source_dim))

    def get_save_params(self):
        """
        returns a dictionary mapping from strings (content names) to arraylike content (ndarray, int, float...).
        This dictionary is the content to be saved.
        :return:
        """
        save_params = SavedObject.get_save_params(self)
        save_params['hyperplane_matrix'] = self.hyperplane_matrix
        if self.means is not None:
            save_params['means'] = self.means
        if self.seed is not None:
            save_params['seed'] = self.seed
        return save_params

    def load_from_params(self, params):
        rewrite = SavedObject.load_from_params(self, params)
        self.hyperplane_matrix = params['hyperplane_matrix']
        assert self.hyperplane_matrix.shape == (self.dest_dim, self.source_dim)
        if 'means' in params:
            self.means = params['means']
            assert self.means.shape == (self.source_dim,)
        else:
            self.means = None
        if 'seed' in params:
            self.seed = params['seed']
        return rewrite


class SpeedyHyperplaneEmbedding(BinaryEmbedding):
    """
    This class implements an asymptotically-faster approximation of hyperplane embedding using Hadamard transforms.
    """

    def __init__(self, prrot_iters=3, **kwargs):
        super().__init__(**kwargs)
        self.prrot_iters = prrot_iters
        self.means = None

        self.prrot_flips = None

        self.rotation_dim = max(self.source_dim, self.dest_dim)
        # verify that the rotation_dim is a power of 2
        self.rotation_dim = 2 ** (int(np.ceil(np.log2(self.rotation_dim))))

        if 'seed' in kwargs:
            self.seed = kwargs['seed']
        else:
            self.seed = None

    def apply(self, words):
        normed_source = words.copy()
        if self.means is not None:
            normed_source -= self.means

        hyper_res = self.rotate_words(normed_source)
        return pack_point((hyper_res < 0).astype(np.int8), self.dest_dim)

    def rotate_words(self, words):
        if self.source_dim < self.rotation_dim:
            # pad by zeros
            pad_list = [(0, 0)] * words.ndim
            pad_list[-1] = (0, self.rotation_dim - self.source_dim)
            words = np.pad(words, pad_width=pad_list, mode='constant')

        for i in range(self.prrot_iters):
            words *= self.prrot_flips[i]
            words = self.hadamard_transform(words)

        return words[..., :self.dest_dim]

    @classmethod
    def hadamard_transform(cls, vector):
        new_vector = np.transpose(vector).copy()
        hadamard_transform_int(new_vector)
        return np.transpose(new_vector).copy()

    def generate(self, means=None):
        super().generate()
        rng = np.random.default_rng(seed=self.seed)
        if means is not None:
            assert means.shape == (self.source_dim,)
            self.means = means
        self.prrot_flips = 1 - 2 * rng.integers(2, size=(self.prrot_iters, self.rotation_dim))

    def get_save_params(self) -> dict:
        save_params = super().get_save_params()
        save_params['prrot_flips'] = self.prrot_flips
        if self.means is not None:
            save_params['means'] = self.means
        if self.seed is not None:
            save_params['seed'] = self.seed
        return save_params

    def load_from_params(self, params):
        rewrite = super().load_from_params(params)
        self.prrot_flips = params['prrot_flips']
        assert self.prrot_flips.shape == (self.prrot_iters, self.dest_dim)
        if 'means' in params:
            self.means = params['means']
            assert self.means.shape == (self.source_dim,)
        else:
            self.means = None
        if 'seed' in params:
            assert self.seed == params['seed']
        return rewrite


@numba.njit
def hadamard_transform_int(vector):
    v_dim = vector.shape[0]
    if v_dim == 1:
        return
    flip = 1
    while flip < v_dim:
        cur = 0
        while cur < v_dim:
            vector[cur:cur + flip, ...] += vector[cur + flip:cur + 2 * flip, ...]
            vector[cur + flip:cur + 2 * flip, ...] = vector[cur:cur + flip, ...] - 2 * vector[cur + flip:cur + 2 * flip,
                                                                                       ...]
            cur += 2 * flip
        flip *= 2
