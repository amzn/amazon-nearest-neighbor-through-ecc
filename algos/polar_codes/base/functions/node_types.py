# Original Copyright 2018 by Grigory Timofeev, MIT License.
# Modifications Copyright 2022 by Amazon.com, Inc. or its affiliates, CC-BY-NC 4.0 License.

import numpy as np


class NodeTypes:
    """Types of decoding nodes."""
    ZERO = 'ZERO'
    ONE = 'ONE'
    SINGLE_ONE = 'SINGLE_ONE'
    SINGLE_ZERO = 'SINGLE_ZERO'
    REPETITION = 'REPETITION'

    OTHER = 'OTHER'


class NodeTypeDetector:
    """Class used to detect the type of decoding node."""
    REPETITION_MIN_SIZE = 2

    def __init__(self, *args, **kwargs):
        self.last_chunk_type = None
        self.mask_steps = None

    def __call__(
            self,
            supported_nodes: list,
            mask: np.array,
            AF: int = 0,
    ) -> str:
        """Get type of decoding Node."""
        self.N = mask.size
        self.AF = AF

        if (NodeTypes.ONE in supported_nodes
                and self._is_one(mask)):
            return NodeTypes.ONE
        if (NodeTypes.SINGLE_ONE in supported_nodes
                and self._is_one(mask)
                and len(mask) == 1):
            return NodeTypes.SINGLE_ONE
        if (NodeTypes.ZERO in supported_nodes
                and self._is_zero(mask)):
            return NodeTypes.ZERO
        if (NodeTypes.SINGLE_ZERO in supported_nodes
                and self._is_zero(mask)
                and len(mask) == 1):
            return NodeTypes.SINGLE_ZERO
        if (NodeTypes.REPETITION in supported_nodes
                and self._is_repetition(mask)):
            return NodeTypes.REPETITION

        return NodeTypes.OTHER

    def _is_one(self, mask: np.array) -> bool:
        return np.all(mask == 1)

    def _is_zero(self, mask: np.array) -> bool:
        return np.all(mask == 0)

    def _is_repetition(self, mask: np.array) -> bool:
        return (
                mask.size >= self.REPETITION_MIN_SIZE and
                mask[-1] == 1 and
                np.sum(mask) == 1
        )


get_node_type = NodeTypeDetector()
