import numpy as np
from ..utils import get_dist
from ..base import BaseDecoder, PolarEncoder


class ExhaustiveDecoder(BaseDecoder):
    ENCODER_CLASS = PolarEncoder

    def __init__(self, *, L, K, N, **kwargs):
        super().__init__(**kwargs)
        self.L = L
        self.K = K
        self.n = int(np.log2(N))
        self.encoder = self.ENCODER_CLASS(mask=self.mask, n=self.n, is_systematic=False)
        self.codebook = self.generate_codebook()

    def generate_codebook(self):
        indices = np.arange(2 ** self.K, dtype=np.uint32)
        byte_view = indices.view(dtype=np.dtype((np.uint8, 4)))
        bit_array = np.unpackbits(byte_view, axis=-1)[..., :self.K]
        return self.encoder.encode(bit_array)

    @classmethod
    def slice(cls, received_llr):
        return (received_llr < 0).astype(np.int8)

    def decode_internal(self, received_llr: np.array) -> np.array:
        to_decode = self.slice(received_llr)
        output = np.zeros(shape=to_decode.shape[:-1] + (self.L, self.N), dtype=np.int8)
        for ind in np.ndindex(to_decode.shape[:-1]):
            distances = get_dist(to_decode[ind], self.codebook)
            part_inds = np.argpartition(distances, self.L - 1, axis=-1)
            best_codewords = np.take_along_axis(self.codebook, part_inds[..., :self.L, np.newaxis], axis=-2)
            output[ind + (slice(None), slice(None))] = best_codewords
        return output
