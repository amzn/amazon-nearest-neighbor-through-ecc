# Original Copyright 2018 by Grigory Timofeev, MIT License.
# Modifications Copyright 2022 by Amazon.com, Inc. or its affiliates, CC-BY-NC 4.0 License.

import abc

import numpy as np
from ...utils.standard_utils import unpack_point, pack_point
from .. import utils
# from ..utils import get_llrs

from . import encoder


class BasePolarCodec(metaclass=abc.ABCMeta):
    """Basic codec for Polar code.

    Includes code construction.
    Defines the basic workflow for encoding and decoding.

    Supports creation of a polar code from custom mask.

    """
    encoder_class = encoder.PolarEncoder
    decoder_class = None

    def __init__(self, *, N: int, K: int,
                 is_systematic: bool = True,
                 mask,
                 ):

        assert K <= N, (f'Cannot create Polar code with N = {N}, K = {K}.'
                        f'\nN must be at least K.')

        self.N = N
        self.K = K
        self.n = int(np.log2(N))
        self.is_systematic = is_systematic
        self.mask = mask
        self.encoder = self.init_encoder()
        self.decoder = self.init_decoder()

    def __str__(self):
        return (f'({self.N}, {self.K}) Polar code.\n'
                f'Systematic: {str(self.is_systematic)}\n')

    def to_dict(self):
        """Get code parameters as a dict."""
        return {
            'type': self.__class__.__name__,
            'N': self.N,
            'K': self.K,
            'is_systematic': self.is_systematic,
            'mask': self.mask.copy()
        }

    def init_encoder(self):
        """Get Polar Encoder instance."""
        return self.encoder_class(mask=self.mask, n=self.n,
                                  is_systematic=self.is_systematic)

    @abc.abstractmethod
    def init_decoder(self):
        """Get Polar Decoder instance."""

    def encode(self, message: np.array) -> np.array:
        """Encode binary message."""
        return self.encoder.encode(message)

    def decode(self, received_message: np.array, to_info=True) -> np.array:
        """Decode received message presented as LLR values."""
        return self.decoder.decode(received_message, to_info=to_info)

    def simple_decode(self, words, to_info=True, packed=True):
        if packed:
            words = unpack_point(words, self.N)

        to_decode = utils.get_llrs(words)
        result = self.decode(to_decode, to_info=to_info)
        result_dim = (self.K if to_info else self.N)
        if packed:
            result = pack_point(result, result_dim)
        return result
