from .exhaustive_decoder import ExhaustiveDecoder

from ..base import BasePolarCodec


class ExhaustiveCodec(BasePolarCodec):
    """Polar code with SC List decoding algorithm."""
    decoder_class = ExhaustiveDecoder

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
        return self.decoder_class(n=self.n, K=self.K, N=self.N, mask=self.mask,
                                  is_systematic=self.is_systematic, L=self.L)
