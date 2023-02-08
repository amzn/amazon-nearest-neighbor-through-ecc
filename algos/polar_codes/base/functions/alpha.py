# Original Copyright 2018 by Grigory Timofeev, MIT License.
# Modifications Copyright 2022 by Amazon.com, Inc. or its affiliates, CC-BY-NC 4.0 License.

import numpy as np


def compute_alpha(a: np.array, b: np.array) -> np.array:
    """Basic function to compute intermediate LLR values."""
    c = np.abs(a)
    np.minimum(c, np.abs(b), out=c)
    c *= np.sign(a * b)
    return c


def compute_left_alpha(llr: np.array) -> np.array:
    """Compute Alpha for left node during SC-based decoding."""
    N = llr.shape[-1] // 2
    left = llr[..., :N]
    right = llr[..., N:]
    return compute_alpha(left, right)


def compute_right_alpha(llr: np.array,
                        left_beta: np.array) -> np.array:
    """Compute Alpha for right node during SC-based decoding."""
    N = llr.shape[-1] // 2
    left = llr[..., :N]
    right = llr[..., N:]
    return right - (2 * left_beta[..., slice(None)] - 1) * left
