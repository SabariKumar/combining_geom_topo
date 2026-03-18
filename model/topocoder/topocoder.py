# PyTorch implementation of TopoCoder, based on RipsNet (de Surrel et. al., PMLR 196:96-106, 2022.)

from functools import partial
from typing import List

import torch
import torch.nn as nn

from model.topocoder.topocoder_utils import DeepSetLayer, DeepSetSum


class TopoCoder(nn.Module):
    # Model is a composition of several permutation invariant DeepSets layers,
    # followed by node pooling via DeepSetSum and transformation via sequence of
    # dense layers.

    def __init__(
        self,
        input_shape: int,
        output_shape: int,
        deepsets_shapes: List[int],
        dense_shapes: List[int],
        use_bias: bool = True,
        use_sigmoid=True,
    ):
        super().__init__()
        deepsets_shapes = [input_shape] + deepsets_shapes
        dense_shapes = dense_shapes + [output_shape]
        # The sum function flattens the list of layer tuples into a list of layers
        layers = list(
            sum(
                [
                    (DeepSetLayer(x[0], x[1], use_bias=use_bias), nn.ReLU())
                    for x in zip(deepsets_shapes[:-1], deepsets_shapes[1:])
                ],
                (),
            )
        )
        layers += [DeepSetSum(deepsets_shapes[-1])]
        layers += list(
            sum(
                [
                    (nn.Linear(x[0], x[1]), nn.ReLU())
                    for x in zip(dense_shapes[:-1], dense_shapes[1:])
                ],
                (),
            )
        )
        if use_sigmoid:
            layers += [nn.Sigmoid()]
        self.topo_net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.topo_net(x)
        return out


# Predefined model instances match paper implementations. Note that due to use of functools.partial,
# models defined here can't be subclassed. Use TopoCoder base class instead.
# small_TopoCoder = partial(
#     TopoCoder(
#         deepsets_shapes=[100, 30, 20, 10],
#         dense_shapes=[10, 50, 100],
#         use_bias=True,
#         use_sigmoid=True,
#     )
# )
# big_TopoCoder = partial(
#     TopoCoder(
#         deepsets_shapes=[1200, 600, 300, 150, 50],
#         dense_shapes=[50, 75, 100],
#         use_bias=True,
#         use_sigmoid=True,
#     )
# )
