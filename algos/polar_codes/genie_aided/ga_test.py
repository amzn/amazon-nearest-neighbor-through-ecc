import sys

import numpy as np
from ..list_codecs import PolarDecoderSingle
from ..utils import get_llrs
import multiprocessing as mp
from ..base import PolarEncoder


class GenieAidedTest:
    DECODER_CLASS = PolarDecoderSingle
    ENCODER_CLASS = PolarEncoder

    def __init__(self, N):
        self.n = int(np.log2(N))
        self.N = N
        self.decoder = self.DECODER_CLASS(n=self.n, mask=np.ones(N), L=1)
        self.encoder = self.ENCODER_CLASS(mask=np.ones(N), n=self.n, is_systematic=False)

    def genie_aided(self, words, noisy_words, verbose=False, num_proc=None):
        """
        This function performs genie-aided code mask generation: given a set of code words and their noisy versions,
        it evaluates the noise on each virtual data channel to find the best channels.
        :param words:
        :param noisy_words:
        :param verbose:
        :param num_proc:
        :return: the error count on each virtual data channel.
        """
        if num_proc is None:
            trace = sys.gettrace()
            if trace is None:
                num_proc = mp.cpu_count()
            else:
                num_proc = 1
        processes = []
        result_queue = mp.Queue()
        for i in range(num_proc):
            proc_slice = slice(i, len(words), num_proc)
            process_args = [words[proc_slice].copy(), noisy_words[proc_slice].copy(), result_queue]
            processes.append(mp.Process(target=self.genie_aided_sp,
                                        args=process_args))
        for i in range(num_proc):
            processes[i].start()

        errs = np.zeros(self.N, dtype=int)
        for i in range(num_proc):
            errs += result_queue.get()

        if verbose:
            best_50 = np.sort(errs)[:50]
            print(f'processed all words, best 50 bits have these errors: {best_50}')
        return errs

    def genie_aided_sp(self, words, noisy_words, queue):
        batch_size = 50000
        result = np.zeros(self.N, dtype=int)
        for i in range(0, len(words), batch_size):
            start_ind = i
            end_ind = min(i + batch_size, len(words))
            batch_slice = slice(start_ind, end_ind)
            result += self.genie_aided_batch(words[batch_slice], noisy_words[batch_slice])
        queue.put(result)

    def genie_aided_batch(self, words, noisy_words):
        result = np.zeros(self.N, dtype=int)
        assert words.shape == noisy_words.shape
        data_words = self.encoder.encode(words)
        llrs = get_llrs(noisy_words)
        self.decoder._set_initial_state(llrs)
        assert len(self.decoder.leaves) == self.N

        for leaf_ind in range(self.N):
            leaf = self.decoder.leaves[leaf_ind]
            self.decoder.cur_leaf_ind = leaf_ind
            self.decoder._compute_intermediate_alpha(leaf)
            self.decoder._decode_leaf(leaf)

            # count errors and fix beta
            result[leaf_ind] += np.count_nonzero(leaf.beta[:, 0, 0] ^ data_words[:, leaf_ind])
            leaf.beta[..., 0, 0] = data_words[:, leaf_ind]

            self.decoder._compute_intermediate_beta(leaf)
            self.decoder.set_decoder_state(self.decoder._position)
        return result
