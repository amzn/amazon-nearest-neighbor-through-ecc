# Original Copyright 2018 by Grigory Timofeev, MIT License.
# Modifications Copyright 2022 by Amazon.com, Inc. or its affiliates, CC-BY-NC 4.0 License.


from ..base import (
    BaseDecoder
, compute_right_alpha
, compute_left_alpha
, compute_parent_beta_hard
)
from src.algos.polar_codes.list_codecs.node import (
    PolarNode,
    PolarNodeSingle
)

import numpy as np
from anytree import PreOrderIter
from ..base.functions.node_types import NodeTypes

import multiprocessing

from threading import Thread


class PolarDecoder(BaseDecoder):
    node_class = PolarNode

    def __init__(self, n: int,
                 mask: np.array,
                 L: int = 1):

        super().__init__(n=n, mask=mask, is_systematic=True)
        # self.N = mask.shape[0]
        # self.n = n
        # self.mask = mask

        self._decoding_tree = self._setup_decoding_tree()
        self._position = 0
        self._leaves = None
        self.L = L
        self.num_paths = None
        self.path_metrics = None
        self.data_shape = None
        self.cur_leaf_ind = None
        # self.subdec = self.SUBDEC_CLASS(n=n, mask=mask)

    def _set_initial_state(self, received_llr):
        """Initialize paths with received message."""
        self._position = 0
        self.num_paths = 1
        self.data_shape = received_llr.shape[:-1]

        # the llr that goes to the root is of the maximum list length, since it doesn't change from now on.
        root_llr = received_llr[..., np.newaxis, :].copy()
        self.path_metrics = np.zeros(self.data_shape + (1,), dtype=int)
        self._reset_tree_computed_state()
        self._decoding_tree.root.alpha = root_llr

    def _reset_tree_computed_state(self):
        """Reset the state of the tree before decoding"""
        for node in PreOrderIter(self._decoding_tree):
            node.is_computed = False
            node.choices = None
            node.beta = node.alpha = None

    def set_decoder_state(self, position):
        """Set current state of each path."""
        self._position = position

    def _compute_intermediate_alpha(self, leaf):
        """Compute intermediate Alpha values (LLR)."""
        for node in leaf.path[1:]:
            if node.is_computed:
                continue

            parent = node.parent
            parent_alpha = parent.alpha

            if node.is_left:
                node.alpha = compute_left_alpha(parent_alpha)

            else:  # node is right child
                left_node = node.siblings[0]
                left_beta = left_node.beta
                alpha_type = np.dtype([('alpha', int, parent.N)])
                parent_alpha_view = parent_alpha.view(dtype=alpha_type)
                chosen_parent_alpha_view = np.take_along_axis(parent_alpha_view, left_node.choices[..., np.newaxis],
                                                              axis=-2)
                chosen_parent_alpha = chosen_parent_alpha_view.view(int)
                node.alpha = compute_right_alpha(chosen_parent_alpha, left_beta)

            node.is_computed = True

    def _compute_intermediate_beta(self, node):
        """Compute intermediate Beta values (BIT)."""
        if node.is_left:
            return

        if node.is_root:
            return

        parent = node.parent
        left = node.siblings[0]
        # padded_node_choices = np.repeat(node.choices[..., np.newaxis], left.N, axis=-1)
        beta_type = np.dtype([('beta', np.int8, left.N)])
        left_beta_view = left.beta.view(beta_type)

        chosen_left_beta_view = np.take_along_axis(left_beta_view, node.choices[..., np.newaxis], axis=-2)
        chosen_left_beta = chosen_left_beta_view.view(np.int8)
        parent.beta = compute_parent_beta_hard(chosen_left_beta, node.beta)
        parent.choices = np.take_along_axis(left.choices, node.choices, axis=-1)

        # zero unnecessary fields to save memory
        node.beta = node.alpha = node.choices = None
        left.beta = left.alpha = left.choices = None

        return self._compute_intermediate_beta(parent)

    def adopt_paths(self, cur_num_paths, cur_path_inds, cur_betas, cur_metrics):
        self.path_metrics = cur_metrics
        self.num_paths = cur_num_paths
        leaf = self.leaves[self.cur_leaf_ind]
        leaf.beta = cur_betas.copy()
        leaf.choices = cur_path_inds

    # have to override the 'zero' node function to fix the choices
    def _decode_leaf_zero(self, leaf):
        cur_betas = np.zeros(self.data_shape + (self.num_paths, leaf.N), dtype=np.int8)

        cur_path_inds = np.zeros(self.data_shape + (self.num_paths,), dtype=int)
        # let's try this
        alpha_scores = np.sum(np.minimum(leaf.alpha, 0), axis=-1)
        cur_metrics = self.path_metrics - np.abs(alpha_scores)
        cur_path_inds[..., :] = np.arange(self.num_paths)
        self.path_metrics -= np.abs(alpha_scores)
        self.adopt_paths(self.num_paths, cur_path_inds, cur_betas, cur_metrics)

    def _decode_leaf_one(self, leaf):
        cur_num_paths = self.num_paths
        path_type = np.dtype([
            ('metric', int),
            ('index', int),
            ('beta', np.int8, (leaf.N,))
        ])

        cur_paths = np.zeros(self.data_shape + (cur_num_paths,), dtype=path_type)
        cur_paths['index'] = np.arange(cur_num_paths)
        cur_paths['metric'] = self.path_metrics

        for i in range(leaf.N):
            # compute alphas for this iteration.
            # can we do this better?

            cur_bit_alpha = np.take_along_axis(leaf.alpha[..., i], cur_paths['index'], axis=-1)

            # generate new arrays
            new_num_paths = 2 * cur_num_paths
            new_paths = np.zeros(self.data_shape + (new_num_paths,), dtype=path_type)

            # add zero choices
            new_paths[..., :cur_num_paths]['beta'] = cur_paths['beta']
            new_paths[..., :cur_num_paths]['metric'] = cur_paths['metric'] - (
                        np.abs(np.sign(cur_bit_alpha) - 1) * np.abs(cur_bit_alpha)) // 2
            new_paths[..., :cur_num_paths]['index'] = cur_paths['index']

            # add one choices
            new_paths[..., cur_num_paths:]['beta'] = cur_paths['beta']
            new_paths['beta'][..., cur_num_paths:, i] = 1
            new_paths[..., cur_num_paths:]['metric'] = cur_paths['metric'] - (
                        np.abs(np.sign(cur_bit_alpha) + 1) * np.abs(cur_bit_alpha)) // 2
            new_paths[..., cur_num_paths:]['index'] = cur_paths['index']

            # prune choices
            pruned_num_paths = min(new_num_paths, self.L)
            # partition in descending order through flipping metrics
            new_paths['metric'] *= -1
            sort_inds = np.argpartition(new_paths['metric'], pruned_num_paths - 1, axis=-1)
            cur_paths = np.take_along_axis(new_paths, sort_inds[..., :pruned_num_paths], axis=-1)
            cur_paths['metric'] *= -1
            cur_num_paths = pruned_num_paths

        self.adopt_paths(cur_num_paths, cur_paths['index'], cur_paths['beta'], cur_paths['metric'])

    def _decode_leaf_rep(self, leaf):

        cur_num_paths = self.num_paths
        path_type = np.dtype([
            ('metric', int),
            ('index', int),
            ('beta', np.int8, leaf.N)
        ])

        # generate new arrays
        new_num_paths = 2 * cur_num_paths
        new_paths = np.zeros(self.data_shape + (new_num_paths,), dtype=path_type)
        new_paths[..., :cur_num_paths]['metric'] = new_paths[..., cur_num_paths:]['metric'] = self.path_metrics
        new_paths[..., :cur_num_paths]['index'] = new_paths[..., cur_num_paths:]['index'] = np.arange(cur_num_paths)
        new_paths[..., cur_num_paths:]['beta'] = 1

        # calc alpha penalties
        zero_penalty = -np.sum(np.minimum(leaf.alpha, 0), axis=-1)
        one_penalty = np.sum(np.maximum(leaf.alpha, 0), axis=-1)

        new_paths[..., :cur_num_paths]['metric'] -= zero_penalty
        new_paths[..., cur_num_paths:]['metric'] -= one_penalty

        # prune choices
        pruned_num_paths = min(new_num_paths, self.L)
        new_paths['metric'] *= -1
        sort_inds = np.argpartition(new_paths['metric'], pruned_num_paths - 1, axis=-1)
        cur_paths = np.take_along_axis(new_paths, sort_inds[..., :pruned_num_paths], axis=-1)
        cur_paths['metric'] *= -1
        cur_num_paths = pruned_num_paths
        self.adopt_paths(cur_num_paths, cur_paths['index'], cur_paths['beta'].copy(), cur_paths['metric'])

    def set_L(self, L):
        self.L = L

    def decode_internal(self, received_llr: np.array) -> np.array:
        self._set_initial_state(received_llr)

        for leaf_ind in range(len(self.leaves)):
            leaf = self.leaves[leaf_ind]
            self.cur_leaf_ind = leaf_ind
            self._compute_intermediate_alpha(leaf)
            self._decode_leaf(leaf)
            self._compute_intermediate_beta(leaf)
            self.set_decoder_state(self._position + leaf.N)

        # sort according to score
        sort_inds = np.argsort(self.path_metrics, axis=-1)
        flipped_sort_inds = np.flip(sort_inds, axis=-1)
        sort_inds = flipped_sort_inds
        self.path_metrics = np.take_along_axis(self.path_metrics, sort_inds, axis=-1)
        root = self.root

        beta_type = np.dtype([('beta', np.int8, root.N)])
        root_beta_view = root.beta.view(beta_type)
        result_view = np.take_along_axis(root_beta_view, sort_inds[..., np.newaxis], axis=-2)
        self.result = result_view.view(np.int8)
        return self.result

    def _decode_leaf(self, leaf):

        if leaf.node_type == NodeTypes.ZERO \
                or leaf.node_type == NodeTypes.SINGLE_ZERO:
            self._decode_leaf_zero(leaf)
        elif leaf.node_type == NodeTypes.ONE \
                or leaf.node_type == NodeTypes.SINGLE_ONE:
            self._decode_leaf_one(leaf)
        elif leaf.node_type == NodeTypes.REPETITION:
            self._decode_leaf_rep(leaf)
        else:
            raise Exception('This type of node is not implemented')

    def _setup_decoding_tree(self, ):
        """Setup decoding tree."""
        return self.node_class(mask=self.mask)

    @property
    def leaves(self):
        if self._leaves is None:
            self._leaves = self._decoding_tree.leaves

        return self._leaves

    @property
    def root(self):
        """Returns root node of decoding tree."""
        return self._decoding_tree.root


class PolarDecoderSingle(PolarDecoder):
    node_class = PolarNodeSingle


class ParallelListDecoder(BaseDecoder):
    DECODER_CLASS = PolarDecoder
    MAX_BLOCK_SIZE = 10000

    def __init__(self, n: int,
                 mask: np.array,
                 # is_systematic: bool = True,
                 L: int = 1,
                 num_threads=multiprocessing.cpu_count()):
        super().__init__(n=n, mask=mask)
        self.L = L
        self.num_paths = None
        self.path_metrics = None
        self.data_shape = None
        self.num_threads = num_threads
        self.decoders = [self.DECODER_CLASS(n=self.n,
                                            mask=self.mask,
                                            L=self.L) for i in range(self.num_threads)]

    class DecodingThread(Thread):
        def __init__(self, *, thread_index, decoder, num_threads, block_size, words_to_decode, to_info, output):
            super().__init__()
            self.thread_index = thread_index
            self.num_threads = num_threads
            self.block_size = block_size
            self.words_to_decode = words_to_decode
            self.to_info = to_info
            self.decoder = decoder
            self.output = output

        def run(self):
            jump_size = self.block_size * self.num_threads
            starting_position = self.block_size * self.thread_index
            for start_index in range(starting_position, len(self.words_to_decode), jump_size):
                end_index = min(start_index + self.block_size, len(self.words_to_decode))
                chunk_to_decode = self.words_to_decode[start_index:end_index]
                self.output[start_index:end_index] = self.decoder.decode(chunk_to_decode, to_info=self.to_info)

    def decode(self, received_llr: np.array, to_info=True) -> np.array:
        threads = []
        K = np.count_nonzero(self.mask)
        block_size = max(1, min(self.MAX_BLOCK_SIZE, len(received_llr) // self.num_threads))

        if to_info:
            output = np.zeros(shape=(len(received_llr), self.L, K), dtype=np.int8)
        else:
            output = np.zeros(shape=(len(received_llr), self.L, self.N), dtype=np.int8)

        for thread_index in range(self.num_threads):
            threads.append(self.DecodingThread(thread_index=thread_index,
                                               num_threads=self.num_threads,
                                               block_size=block_size,
                                               words_to_decode=received_llr,
                                               to_info=to_info,
                                               decoder=self.decoders[thread_index],
                                               output=output))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        return output

    def decode_internal(self, received_llr: np.array) -> np.array:
        raise Exception('decode_internal not implemented here, use decode instead')
