# Original Copyright 2018 by Grigory Timofeev, MIT License.
# Modifications Copyright 2022 by Amazon.com, Inc. or its affiliates, CC-BY-NC 4.0 License.

from typing import Dict

from src.algos.polar_codes.base import NodeTypes

from src.algos.polar_codes.base.node import BaseNode


class PolarNode(BaseNode):
    supported_nodes = (
        NodeTypes.SINGLE_ZERO,
        NodeTypes.ZERO,
        NodeTypes.ONE,
        NodeTypes.REPETITION,
        NodeTypes.SINGLE_ONE
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.choices = None

    @property
    def is_zero(self) -> bool:
        """Check is the node is Zero node."""
        return self.node_type == NodeTypes.ZERO

    def get_decoding_params(self) -> Dict:
        return dict(
            node_type=self.node_type,
            llr=self.alpha,
        )


class PolarNodeSingle(PolarNode):
    supported_nodes = (
        NodeTypes.SINGLE_ZERO,
        NodeTypes.SINGLE_ONE
    )
