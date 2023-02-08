# Original Copyright 2018 by Grigory Timofeev, MIT License.
# Modifications Copyright 2022 by Amazon.com, Inc. or its affiliates, CC-BY-NC 4.0 License.

import numpy as np


class PolarEncoder:
    """Polar Codes encoder."""

    def __init__(self,
                 mask: np.array,
                 n: int,
                 is_systematic: bool = True):

        self.n = n
        self.N = mask.shape[0]
        self.mask = mask
        self.is_systematic = is_systematic

    def encode(self, message: np.array) -> np.array:
        """Encode message with a polar code.

        Support both non-systematic and systematic encoding.

        """
        precoded = self._precode(message)
        encoded = self._non_systematic_encode(precoded, self.n)

        if self.is_systematic:
            encoded *= self.mask
            encoded = self._non_systematic_encode(encoded, self.n)

        return encoded

    def _precode(self, message: np.array) -> np.array:
        """Apply polar code mask to information message.

        Replace 1's of polar code mask with bits of information message.

        """
        precoded = np.zeros(shape=message.shape[:-1] + (self.N,), dtype=np.int8)
        precoded[..., self.mask == 1] = message
        return precoded

    @staticmethod
    def _non_systematic_encode(message: np.array, n: int) -> np.array:
        """Non-systematic encoding.

        Args:
            message (numpy.array): precoded message to encode.

        Returns:
            message (numpy.array): non-systematically encoded message.

        """
        N = 2 ** n
        num_range = np.arange(N)
        for i in range(n):
            indices = num_range & (1 << i)
            message[..., indices == 0] ^= message[..., indices != 0]
        return message
