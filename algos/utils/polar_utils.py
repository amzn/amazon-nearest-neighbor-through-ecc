from ..polar_codes import PolarCodec
from .standard_utils import *


class PolarUtils:
    """
    This class provides an interface for (list-)decoding of polar codes.
    """
    BASE_LLR_MULT = 3
    LIST_CODEC_CLASS = PolarCodec
    DEFAULT_BATCH_SIZE = 160000

    @classmethod
    def get_llrs(cls, point):
        return cls.BASE_LLR_MULT * (np.ones_like(point, dtype=int) - 2 * point)

    @classmethod
    def decode_polar_list(cls, points_to_decode, *,
                          list_codec,
                          crop=None,
                          xor_mask=None,
                          permutation=None,
                          batch_size=None):
        N = list_codec.N
        K = list_codec.K
        if crop is None:
            crop = list_codec.L
        if xor_mask is None:
            xor_mask = pack_point(np.zeros(N, dtype=np.int8), N)
        if permutation is None:
            permutation = np.arange(N, dtype=int)

        # plan for entire array to take no more than 10GB in RAM
        if batch_size is None:
            batch_size = cls.DEFAULT_BATCH_SIZE

        result = np.ndarray(shape=(points_to_decode.shape[0], crop, math.ceil(K / 8)), dtype=np.uint8)
        for i in range(0, len(result), batch_size):
            start = i
            end = min(len(result), i + batch_size)
            result[start:end] = cls.decode_polar_list_batch(points_to_decode[start:end],
                                                            list_codec=list_codec,
                                                            crop=crop,
                                                            xor_mask=xor_mask,
                                                            permutation=permutation)
        return result

    @classmethod
    def decode_polar_list_batch(cls, points, *, list_codec, crop, xor_mask, permutation):
        # apply permutation
        N = list_codec.N
        unpacked = unpack_point(points, N)
        permuted_and_xored = pack_point(np.take(unpacked, permutation, axis=-1), N) ^ xor_mask
        return list_codec.simple_decode(permuted_and_xored)[..., :crop, :]

    @classmethod
    def get_list_size_from_wanted(self, wanted_size):
        """
        A simple rule for choosing the list size for decoding when we want the closest wanted_size codewords to the
        given bin_words.
        """
        if wanted_size == 1:
            return 16
        if wanted_size <= 16:
            return 32
        if wanted_size <= 128:
            return wanted_size * 2
        return wanted_size
