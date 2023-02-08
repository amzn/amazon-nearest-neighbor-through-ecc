# Original Copyright 2018 by Grigory Timofeev, MIT License.
# Modifications Copyright 2022 by Amazon.com, Inc. or its affiliates, CC-BY-NC 4.0 License.


import multiprocessing

import numpy as np

from ..base import BasePolarCodec
from .fast_decoder import PolarDecoder, ParallelListDecoder, PolarDecoderSingle


class PolarCodec(BasePolarCodec):
    """
    This codec implements simplified successive-cancellation list decoding, as described by Hashemi et al.
    """
    decoder_class = PolarDecoder

    def __init__(self, *, N: int, K: int,
                 is_systematic: bool = True,
                 mask,
                 L: int = 1):
        self.L = L
        super().__init__(N=N, K=K,
                         is_systematic=is_systematic,
                         mask=mask,
                         )

    def init_decoder(self):
        return self.decoder_class(self.n,
                                  self.mask,
                                  self.L)

    def to_dict(self):
        d = super().to_dict()
        d.update({'L': self.L})
        return d

    def decode(self, received_llrs: np.ndarray, to_info=True) -> np.ndarray:
        return self.decoder.decode(received_llrs, to_info=to_info)

    def set_L(self, L):
        self.L = L
        self.decoder.set_L(L)


class PolarCodecSingle(PolarCodec):
    """
    A polar codec which does not prune the decoding tree. This is used for genie-aided decoding.
    """
    decoder_class = PolarDecoderSingle


class ParallelCodec(PolarCodec):
    """
    A multithreaded polar codec for increased performance.
    """
    decoder_class = ParallelListDecoder

    def __init__(self, num_threads=multiprocessing.cpu_count(), **kwargs):
        self.num_threads = num_threads
        super().__init__(**kwargs)

    def init_decoder(self):
        return self.decoder_class(self.n,
                                  self.mask,
                                  # self.is_systematic,
                                  self.L,
                                  self.num_threads)
