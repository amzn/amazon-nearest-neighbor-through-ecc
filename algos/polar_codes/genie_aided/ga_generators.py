from ...utils.standard_utils import *
from .ga_test import GenieAidedTest


class GenieAidedGenerators:

    @classmethod
    def bsc_test(cls, dim, num_points, prob1):
        words = generate_binary_points(num_points=num_points, dim=dim)

        noise = generate_biased_binary_points(num_points=num_points, dim=dim, prob1=prob1)
        noisy_words = words ^ noise

        ga_test = GenieAidedTest(N=dim)
        words_unpacked = unpack_point(words, dim)
        noisy_words_unpacked = unpack_point(noisy_words, dim)
        return ga_test.genie_aided(words=words_unpacked, noisy_words=noisy_words_unpacked)

    @classmethod
    def bsc_locate(cls, dim, data_dim, num_points):

        batch_size = int(5e5)
        desired = int(batch_size * 2e-4)
        upper_prob1 = 1 / 2
        lower_prob1 = 0
        while True:
            current_prob1 = (upper_prob1 + lower_prob1) / 2
            # print(f'current prob1={current_prob1}:')
            results = cls.bsc_test(dim=dim, num_points=batch_size, prob1=current_prob1)
            results.sort()
            frame_errors = results[data_dim - 1]
            if frame_errors > 2 * desired:
                upper_prob1 = current_prob1
            elif frame_errors < desired / 2:
                lower_prob1 = current_prob1
            else:
                break
        print(f'located correct BSC error probability: {current_prob1}')
        results = cls.bsc_test(dim=dim, num_points=num_points, prob1=current_prob1)
        return results, current_prob1

    @classmethod
    def ga_sanity_check(cls, mask, dim):
        dim_bits = int(np.log2(dim))
        listy = list(np.nonzero(mask)[0])
        unfrozen_bits = set(listy)
        for unfrozen in unfrozen_bits:
            for bit_num in range(dim_bits):
                new_bit = unfrozen | (1 << bit_num)
                if not new_bit in unfrozen_bits:
                    return False
        return True
