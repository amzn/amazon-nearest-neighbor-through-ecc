# Original Copyright 2018 by Grigory Timofeev, MIT License.
# Modifications Copyright 2022 by Amazon.com, Inc. or its affiliates, CC-BY-NC 4.0 License.

"""Common functions for polar coding."""
import numpy as np

from .node_types import NodeTypes


# -----------------------------------------------------------------------------
# Making hard decisions during the decoding
# -----------------------------------------------------------------------------

def compute_parent_beta_hard(left: np.array,
                             right: np.array) -> np.array:
    """Compute Beta values for parent Node."""
    result = np.concatenate((left ^ right, right), axis=-1, dtype=np.int8)
    return result
