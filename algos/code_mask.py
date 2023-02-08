import logging

import numpy as np

from src.algos.polar_codes import GenieAidedTest
from src.algos.saved_object import SavedObject
from src.algos.utils.standard_utils import generate_binary_points, generate_biased_binary_points, unpack_point


class BSCCodeMask(SavedObject):
    """
    This class represents a polar code mask, and provides methods for genie-aided generation of the mask on a BSC channel.
    It inherits from SavedObject, which allows saving the mask to file (together with supplementary information regarding its creation).
    """

    def __init__(self, N, K, num_points=12000000):
        """
        Initializes the mask.
        :param N: the code dimension
        :param K: the info dimension
        :param num_points: the number of points to use for the genie aided process. More is more accurate, but takes more time.
        """
        super().__init__()
        self.N = N
        self.K = K
        self.num_points = num_points

    def generate(self):
        """
        Generates the mask by calling bsc_locate_and_test.
        """
        super().generate()
        self.errs, self.prob1 = self.bsc_locate_and_test(dim=self.N, data_dim=self.K, num_points=self.num_points)
        self.calc_mask_from_errs()

    def calc_mask_from_errs(self):
        """
        calculates self.mask (i.e., the binary mask) from the bit errors self.errs.
        """
        en_errs = enumerate(self.errs)
        sorted_errs = sorted(en_errs, key=lambda c: c[1] * self.N - c[0])
        data_bits = sorted_errs[:self.K]
        mask = np.zeros(shape=(self.N,), dtype=np.uint8)
        for data_bit in data_bits:
            mask[data_bit[0]] = 1
        err_on_final_bit = data_bits[self.K - 1][1]
        self.mask = mask
        self.fer = err_on_final_bit / self.num_points

    @classmethod
    def bsc_locate_and_test(cls, dim, data_dim, num_points, batch_size=int(5e5), desired_fer=2e-4):
        """
        This function generates the code mask for the desired pair (dim, data_dim), on num_points points,
        with the goal of reaching an error rate of desired_fer on the worst unmasked bit.
        It does this by performing binary search on the noise parameter of the BSC, until the chosen value yields a noise of
        desired_fer on a small batch. Then, the algorithm runs a full batch of num_points points to create the code mask.
        :param dim: the code dimension
        :param data_dim: the data dimension
        :param num_points: the number of points to test
        :param batch_size: the batch for the binary search, should be less than num_points.
        :param desired_fer: the noise level for the noisiest of the data_dim bits.
        :return: an array of the errors of each bit (to be sorted, and the best data_dim bits taken), as well as the error probability chosen for the BSC.
        """
        desired_frame_errors = int(batch_size * desired_fer)
        upper_prob1 = 1 / 2
        lower_prob1 = 0
        while True:
            current_prob1 = (upper_prob1 + lower_prob1) / 2
            # print(f'current prob1={current_prob1}:')
            results = cls.bsc_test(dim=dim, num_points=batch_size, prob1=current_prob1)
            results.sort()
            frame_errors = results[data_dim - 1]
            if frame_errors > 2 * desired_frame_errors:
                upper_prob1 = current_prob1
            elif frame_errors < desired_frame_errors / 2:
                lower_prob1 = current_prob1
            else:
                break
        logging.info(f'located correct BSC error probability: {current_prob1}')
        errs = cls.bsc_test(dim=dim, num_points=num_points, prob1=current_prob1)
        return errs, current_prob1

    @classmethod
    def bsc_test(cls, dim, num_points, prob1):
        words = generate_binary_points(num_points=num_points, dim=dim)

        noise = generate_biased_binary_points(num_points=num_points, dim=dim, prob1=prob1)
        noisy_words = words ^ noise

        ga_test = GenieAidedTest(N=dim)
        words_unpacked = unpack_point(words, dim)
        noisy_words_unpacked = unpack_point(noisy_words, dim)
        errs = ga_test.genie_aided(words=words_unpacked, noisy_words=noisy_words_unpacked)
        return errs

    def get_save_params(self) -> dict:
        save_params = super().get_save_params()
        save_params['errs'] = self.errs
        save_params['mask'] = self.mask
        save_params['prob1'] = self.prob1
        save_params['num_points'] = self.num_points
        save_params['fer'] = self.fer
        return save_params

    def load_from_params(self, params):
        rewrite = super().load_from_params(params)
        self.errs = params['errs']
        self.mask = params['mask']
        self.prob1 = params['prob1']
        assert self.num_points == params['num_points']
        self.num_points = params['num_points']
        self.fer = params['fer']
        return rewrite
