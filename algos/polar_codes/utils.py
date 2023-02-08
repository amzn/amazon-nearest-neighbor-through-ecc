import numpy as np



def get_dist(p1, p2):
    """
    returns the distance between the given bin_words (broadcasting if necessary).
    Note - here, the bin_words are unpacked, i.e. bit vectors.
    """
    p3 = p1 ^ p2
    return np.count_nonzero(p3, axis=-1)


def get_llrs(point, dtype=int):
    BASE_LLR_MULT = 3
    return BASE_LLR_MULT * (np.ones_like(point, dtype=dtype) - 2 * point)

