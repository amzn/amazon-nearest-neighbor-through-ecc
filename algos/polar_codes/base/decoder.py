# Original Copyright 2018 by Grigory Timofeev, MIT License.
# Modifications Copyright 2022 by Amazon.com, Inc. or its affiliates, CC-BY-NC 4.0 License.

import abc

import numpy as np


class BaseDecoder(metaclass=abc.ABCMeta):
    """Basic class for polar decoder."""

    def __init__(self, *, n, mask: np.array, is_systematic: bool = True):
        self.N = mask.shape[0]
        self.n = n
        self.is_systematic = is_systematic
        self.mask = mask

    def decode(self, received_llr: np.array, to_info=True) -> np.array:
        code_word = self.decode_internal(received_llr)
        if to_info:
            return self.extract_result(code_word)
        return code_word

    @abc.abstractmethod
    def decode_internal(self, received_llr: np.array) -> np.array:
        """Implementation of particular decoding method."""

    def extract_result(self, decoded: np.array) -> np.array:
        """Get decoding result.

        Extract info bits from decoded message due to polar code mask.

        """
        return decoded[..., self.mask == 1]
