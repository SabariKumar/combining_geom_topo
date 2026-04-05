# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

import logging
from typing import Optional, Literal, Dict
import warnings
warnings.filterwarnings('error', message='.*requires.*')

import os
import torch
import torch.nn as nn
import dgl # type: ignore
from dgl import DGLGraph # type: ignore
import dgl.sparse as dglsp
from torch import Tensor
from e3nn import o3 # type: ignore
#from e3nn.util.jit import compile_mode # type: ignore
from einops import repeat # type: ignore
import math
import sys
sys.path.append("/home/sabari/ProteinSol/topoformer")

from model.topoformer.basis import get_basis, update_basis_with_fused
from model.topoformer.layers.linear import LinearSE3, DropoutSE3
from model.topoformer.layers.attention import AttentionBlockSE3
from model.topoformer.layers.convolution import ConvSE3, ConvSE3FuseLevel
from model.topoformer.layers.norm import NormSE3
from model.topoformer.layers.pooling import GPooling
from model.topoformer.utils import str2bool, aggregate_residual, set_requires_grad
from model.topoformer.fiber import Fiber, fiber_to_irreps, fiber_dict_to_flat, fiber_dict_from_flat
from model.topoformer.layers.bettiattention import BettiAttention

class Sequential(nn.Sequential):
    """ Sequential module with arbitrary forward args and kwargs. Used to pass graph, basis and edge features. """

    def forward(self, input, *args, **kwargs):
        input = set_requires_grad(input)
        # args = set_requires_grad(args)
        # kwargs = set_requires_grad(kwargs)
        #print(input)
        for module in self:
            input = module(input, *args, **kwargs)
        return input


def get_populated_edge_features(relative_pos: Tensor, edge_features: Optional[Dict[str, Tensor]] = None):
    """ Add relative positions to existing edge features """
    edge_features = edge_features.copy() if edge_features else {}
    r = relative_pos.norm(dim=-1, keepdim=True)
    if '0' in edge_features:
        edge_features['0'] = torch.cat([edge_features['0'], r[..., None]], dim=1)
    else:
        edge_features['0'] = r[..., None]

    return edge_features


class SE3Transformer(nn.Module):
    def __init__(self,
                 num_layers: int,
                 fiber_in: Fiber,
                 fiber_hidden: Fiber,
                 fiber_out: Fiber,
                 num_heads: int,
                 channels_div: int,
                 fiber_edge: Fiber = Fiber({}),
                 return_type: Optional[int] = None,
                 pooling: Optional[Literal['avg', 'max']] = None,
                 norm: bool = True,
                 use_layer_norm: bool = True,
                 tensor_cores: bool = False,
                 low_memory: bool = False,
                 **kwargs):
        """
        :param num_layers:          Number of attention layers
        :param fiber_in:            Input fiber description
        :param fiber_hidden:        Hidden fiber description
        :param fiber_out:           Output fiber description
        :param fiber_edge:          Input edge fiber description
        :param num_heads:           Number of attention heads
        :param channels_div:        Channels division before feeding to attention layer
        :param return_type:         Return only features of this type
        :param pooling:             'avg' or 'max' graph pooling before MLP layers
        :param norm:                Apply a normalization layer after each attention block
        :param use_layer_norm:      Apply layer normalization between MLP layers
        :param tensor_cores:        True if using Tensor Cores (affects the use of fully fused convs, and padded bases)
        :param low_memory:          If True, will use slower ops that use less memory
        """
        super().__init__()
        self.num_layers = num_layers
        self.fiber_edge = fiber_edge
        self.num_heads = num_heads
        self.channels_div = channels_div
        self.return_type = return_type
        self.pooling = pooling
        self.max_degree = max(*fiber_in.degrees, *fiber_hidden.degrees, *fiber_out.degrees)
        self.tensor_cores = tensor_cores
        self.low_memory = low_memory

        if low_memory:
            self.fuse_level = ConvSE3FuseLevel.NONE
        else:
            # Fully fused convolutions when using Tensor Cores (and not low memory mode)
            self.fuse_level = ConvSE3FuseLevel.FULL if tensor_cores else ConvSE3FuseLevel.PARTIAL

        graph_modules = []
        for i in range(num_layers):
            graph_modules.append(AttentionBlockSE3(fiber_in=fiber_in,
                                                   fiber_out=fiber_hidden,
                                                   fiber_edge=fiber_edge,
                                                   num_heads=num_heads,
                                                   channels_div=channels_div,
                                                   use_layer_norm=use_layer_norm,
                                                   max_degree=self.max_degree,
                                                   fuse_level=self.fuse_level,
                                                   low_memory=low_memory))
            if norm:
                graph_modules.append(NormSE3(fiber_hidden))
            fiber_in = fiber_hidden

        graph_modules.append(ConvSE3(fiber_in=fiber_in,
                                     fiber_out=fiber_out,
                                     fiber_edge=fiber_edge,
                                     self_interaction=True,
                                     use_layer_norm=use_layer_norm,
                                     max_degree=self.max_degree,
                                     fuse_level=self.fuse_level,
                                     low_memory=low_memory))
        self.graph_modules = Sequential(*graph_modules)

        if pooling is not None:
            assert return_type is not None, 'return_type must be specified when pooling'
            self.pooling_module = GPooling(pool=pooling, feat_type=return_type)

    def forward(self, graph: DGLGraph, node_feats: Dict[str, Tensor],
                edge_feats: Optional[Dict[str, Tensor]] = None,
                basis: Optional[Dict[str, Tensor]] = None):
        # Compute bases in case they weren't precomputed as part of the data loading
        basis = basis or get_basis(graph.edata['rel_pos'], max_degree=self.max_degree, compute_gradients=False,
                                   use_pad_trick=self.tensor_cores and not self.low_memory,
                                   amp=torch.is_autocast_enabled())

        # Add fused bases (per output degree, per input degree, and fully fused) to the dict
        basis = update_basis_with_fused(basis, self.max_degree, use_pad_trick=self.tensor_cores and not self.low_memory,
                                        fully_fused=self.fuse_level == ConvSE3FuseLevel.FULL)

        edge_feats = get_populated_edge_features(graph.edata['rel_pos'], edge_feats)

        node_feats = self.graph_modules(node_feats, edge_feats, graph=graph, basis=basis)

        if self.pooling is not None:
            return self.pooling_module(node_feats, graph=graph)

        if self.return_type is not None:
            return node_feats[str(self.return_type)]

        return node_feats

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--num_layers', type=int, default=7,
                            help='Number of stacked Transformer layers')
        parser.add_argument('--num_heads', type=int, default=8,
                            help='Number of heads in self-attention')
        parser.add_argument('--channels_div', type=int, default=2,
                            help='Channels division before feeding to attention layer')
        parser.add_argument('--pooling', type=str, default=None, const=None, nargs='?', choices=['max', 'avg'],
                            help='Type of graph pooling')
        parser.add_argument('--norm', type=str2bool, nargs='?', const=True, default=False,
                            help='Apply a normalization layer after each attention block')
        parser.add_argument('--use_layer_norm', type=str2bool, nargs='?', const=True, default=False,
                            help='Apply layer normalization between MLP layers')
        parser.add_argument('--low_memory', type=str2bool, nargs='?', const=True, default=False,
                            help='If true, will use fused ops that are slower but that use less memory '
                                 '(expect 25 percent less memory). '
                                 'Only has an effect if AMP is enabled on Volta GPUs, or if running on Ampere GPUs')

        return parser


class SE3TransformerPooled(nn.Module):
    def __init__(self,
                 fiber_in: Fiber,
                 fiber_out: Fiber,
                 fiber_edge: Fiber,
                 num_degrees: int,
                 num_channels: int,
                 output_dim: int,
                 **kwargs):
        super().__init__()
        kwargs['pooling'] = kwargs['pooling'] or 'max'
        self.transformer = SE3Transformer(
            fiber_in=fiber_in,
            fiber_hidden=Fiber.create(num_degrees, num_channels),
            fiber_out=fiber_out,
            fiber_edge=fiber_edge,
            return_type=0,
            **kwargs
        )

        n_out_features = fiber_out.num_features
        self.mlp = nn.Sequential(
            nn.Linear(n_out_features, n_out_features),
            nn.ReLU(),
            nn.Linear(n_out_features, output_dim)
        )

    def forward(self, graph, node_feats, edge_feats, basis=None):
        feats = self.transformer(graph, node_feats, edge_feats, basis).squeeze(-1)
        y = self.mlp(feats).squeeze(-1)
        return y

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("Model architecture")
        SE3Transformer.add_argparse_args(parser)
        parser.add_argument('--num_degrees',
                            help='Number of degrees to use. Hidden features will have types [0, ..., num_degrees - 1]',
                            type=int, default=4)
        parser.add_argument('--num_channels', help='Number of channels for the hidden features', type=int, default=32)
        return parent_parser

class TopoformerPooled(nn.Module):
    def __init__(self,
                 fiber_in: Fiber,
                 fiber_out: Fiber,
                 fiber_edge: Fiber,
                 num_degrees: int,
                 num_channels: int,
                 output_dim: int,
                 save_feats_dir: str,
                 run_id: str,
                 **kwargs):
        super().__init__()
        kwargs['pooling'] = kwargs['pooling'] or 'max'
        self.save_feats_dir = save_feats_dir
        self.run_id = run_id
        self.transformer = Topoformer(
            fiber_in = fiber_in,
            fiber_hidden = Fiber.create(num_degrees, num_channels),
            fiber_out = fiber_out,
            fiber_edge = fiber_edge,
            return_type = 0,
            **kwargs
        )
        self.post_pooled_feats = None
        n_out_features = fiber_out.num_features
        dropout = kwargs.get('dropout', 0.3)
        self.mlp = nn.Sequential(
            nn.Linear(n_out_features, n_out_features),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(n_out_features, output_dim)
        )

    def forward(self, graph, node_feats, topo_feats, edge_feats, basis=None):
        feats = self.transformer(graph, node_feats, topo_feats, edge_feats, basis).squeeze(-1)
        y = self.mlp(feats).squeeze(-1)
        return y

    # def save_node_feats(self, pre_pooled_save_file: str, post_pooled_save_file: str):
    #     torch.save(self.transformer.pre_pooled_feats, pre_pooled_save_file)
    #     torch.save(self.post_pooled_feats, post_pooled_save_file)

    #     model.save_node_feats(os.path.join(args.log_dir, f'{run_id}_pre_pooling.pt'),
    #                           os.path.join(args.log_dir, f'{run_id}_post_pooling.pt'))

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("Model architecture")
        SE3Transformer.add_argparse_args(parser)
        parser.add_argument('--num_degrees',
                            help='Number of degrees to use. Hidden features will have types [0, ..., num_degrees - 1]',
                            type=int, default=4)
        parser.add_argument('--num_channels', help='Number of channels for the hidden features', type=int, default=32)
        parser.add_argument('--dropout', help='Dropout probability in the output MLP', type=float, default=0.3)
        return parent_parser

class Topoformer(nn.Module):
    # Composed of stacks of ContractRepsBlocks!

    def __init__(self,
                 num_layers: int,
                 fiber_in: Fiber,
                 fiber_hidden: Fiber,
                 fiber_out: Fiber,
                 num_heads: int,
                 channels_div: int,
                 fiber_edge: Fiber = Fiber({}),
                 return_type: Optional[int] = None,
                 pooling: Optional[Literal['avg', 'max']] = None,
                 norm: bool = True,
                 use_layer_norm: bool = True,
                 tensor_cores: bool = False,
                 low_memory: bool = True,
                 eq_dropout: float = 0.0,
                 comb_type: str = 'fctp', #fctp or conv
                 topo_output_fiber: Fiber = Fiber({'0': 300}),
                 topo_embedding_dim: int = 300,
                 topo_input_fiber: Fiber = Fiber({'0': 100}),
                 topo_num_heads: int = 2,
                 topo_dropout = 0.0,
                 use_topo_projection = False,
                 topo_barcode_input_size = 100,
                 topo_proj_hidden_dim = 600,
                 topo_proj_dim = 300,
                 use_node_mha = True,
                 **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.max_degree = max(*fiber_in.degrees, *fiber_hidden.degrees, *fiber_out.degrees,
                        *topo_input_fiber.degrees, *topo_output_fiber.degrees)
        self.tensor_cores = tensor_cores
        self.low_memory = low_memory
        self.pooling = pooling
        self.return_type = return_type
        self.use_topo_projection = use_topo_projection
        if low_memory:
            self.fuse_level = ConvSE3FuseLevel.NONE
        else:
            # Fully fused convolutions when using Tensor Cores (and not low memory mode)
            self.fuse_level = ConvSE3FuseLevel.FULL if tensor_cores else ConvSE3FuseLevel.PARTIAL


        #Nonlinear projection of topo vectors to reduce memory overhead
        if self.use_topo_projection:
            self.topo_mlp = nn.Sequential(
                nn.Linear(topo_barcode_input_size, topo_proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(topo_proj_hidden_dim, topo_proj_dim)
            )

        contract_modules = []
        contract_modules.append(ContractRepsBlock(
            fiber_in = fiber_in,
            fiber_hidden = fiber_hidden,
            fiber_out = fiber_out,
            max_degree = self.max_degree,
            num_heads = num_heads,
            channels_div = channels_div,
            fiber_edge = fiber_edge,
            return_type = return_type,
            pooling = pooling,
            norm = norm,
            use_layer_norm = use_layer_norm,
            tensor_cores = tensor_cores,
            low_memory = low_memory,
            comb_type = comb_type, #fctp or conv
            topo_output_fiber = topo_output_fiber,
            topo_embedding_dim = topo_embedding_dim,
            topo_input_fiber = topo_input_fiber,
            topo_num_heads = topo_num_heads,
            topo_dropout = topo_dropout,
        ))
        for i in range(num_layers - 1):
            if eq_dropout != 0.0:
                contract_modules.append(DropoutSE3(
                    fiber_in = fiber_out,
                    prob = eq_dropout
                ))
            contract_modules.append(ContractRepsBlock(
                fiber_in = fiber_out,
                fiber_hidden = fiber_hidden,
                fiber_out = fiber_out,
                max_degree = self.max_degree,
                num_heads = num_heads,
                channels_div = channels_div,
                fiber_edge = fiber_edge,
                return_type = return_type,
                pooling = pooling,
                norm = norm,
                use_layer_norm = use_layer_norm,
                tensor_cores = tensor_cores,
                low_memory = low_memory,
                comb_type = comb_type, #fctp or conv
                topo_output_fiber = topo_output_fiber,
                topo_embedding_dim = topo_embedding_dim,
                topo_input_fiber = topo_input_fiber,
                topo_num_heads = topo_num_heads,
                topo_dropout = topo_dropout,
                sum_residual = True,
            ))

        # contract_modules.append(ConvSE3(fiber_in=fiber_out,
        #                              fiber_out=fiber_out,
        #                              fiber_edge=fiber_edge,
        #                              self_interaction=True,
        #                              use_layer_norm=use_layer_norm,
        #                              max_degree=self.max_degree,
        #                              fuse_level=self.fuse_level,
        #                              low_memory=low_memory))
        
        self.contract_modules = Sequential(*contract_modules)

        # if use_node_mha:
            

        if pooling is not None:
            assert return_type is not None, 'return_type must be specified when pooling'
            self.pooling_module = GPooling(pool=pooling, feat_type=return_type)
            self.pre_pooled_feats = None

    def forward(self, graph: DGLGraph, node_feats: Dict[str, Tensor], topo_feats: Dict[str, Tensor],
                edge_feats: Optional[Dict[str, Tensor]] = None,
                basis: Optional[Dict[str, Tensor]] = None):
        
        if self.use_topo_projection:
            topo_feats['0'] = self.topo_mlp(topo_feats['0'])

        # Compute bases in case they weren't precomputed as part of the data loading
        # basis = basis or get_basis(graph.edata['rel_pos'], max_degree=self.max_degree, compute_gradients=False,
        #                            use_pad_trick=self.tensor_cores and not self.low_memory,
        #                            amp=torch.is_autocast_enabled())

        # # Add fused bases (per output degree, per input degree, and fully fused) to the dict
        # basis = update_basis_with_fused(basis, self.max_degree, use_pad_trick=self.tensor_cores and not self.low_memory,
        #                                 fully_fused=self.fuse_level == ConvSE3FuseLevel.FULL)
        edge_feats = get_populated_edge_features(graph.edata['rel_pos'], edge_feats)

        #for module in self.contract_modules:
            #TODO: Add topological stuff, contraction
        node_feats = self.contract_modules(node_feats,
                                           edge_feats,
                                           graph = graph,
                                           basis = basis,
                                           topo_feats = topo_feats)
        if self.pooling is not None:
            self.pre_pooled_feats = node_feats
            return self.pooling_module(node_feats, graph=graph)

        if self.return_type is not None:
            return node_feats[str(self.return_type)]

        return node_feats

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--num_layers', type=int, default=7,
                            help='Number of stacked Transformer layers')
        parser.add_argument('--num_heads', type=int, default=8,
                            help='Number of heads in self-attention')
        parser.add_argument('--channels_div', type=int, default=2,
                            help='Channels division before feeding to attention layer')
        parser.add_argument('--pooling', type=str, default=None, const=None, nargs='?', choices=['max', 'avg'],
                            help='Type of graph pooling')
        parser.add_argument('--norm', type=str2bool, nargs='?', const=True, default=False,
                            help='Apply a normalization layer after each attention block')
        parser.add_argument('--use_layer_norm', type=str2bool, nargs='?', const=True, default=False,
                            help='Apply layer normalization between MLP layers')
        parser.add_argument('--low_memory', type=str2bool, nargs='?', const=True, default=False,
                            help='If true, will use fused ops that are slower but that use less memory '
                                 '(expect 25 percent less memory). '
                                 'Only has an effect if AMP is enabled on Volta GPUs, or if running on Ampere GPUs')

        return parser
    
class ContractRepsBlock(nn.Module):
    # Packages SE(3) Attention, BettiAttention, and the tensor contraction together into a single module
    # On forward, takes topo input and geom input separately, then combines irreps with a choice of e3nn FCTP with
    # learned pooling weights ('fctp') or SE(3)Xformer's convolutional layer ('conv')
    
    def __init__(self,
                 fiber_in: Fiber,
                 fiber_hidden: Fiber,
                 fiber_out: Fiber,
                 max_degree: int,
                 num_heads: int,
                 channels_div: int,
                 fiber_edge: Fiber = Fiber({}),
                 return_type: Optional[int] = None,
                 pooling: Optional[Literal['avg', 'max']] = None,
                 norm: bool = True,
                 use_layer_norm: bool = True,
                 tensor_cores: bool = False,
                 low_memory: bool = True,
                 comb_type: str = 'attn', #fctp or attn
                 topo_output_fiber: Fiber = Fiber({'0': 300}),
                 topo_embedding_dim: int = 300,
                 topo_input_fiber: Fiber = Fiber({'0': 100}),
                 topo_num_heads: int = 2,
                 topo_dropout = 0.0,
                 sum_residual = False,
                 **kwargs):
        
        super().__init__()

        self.comb_type = comb_type
        self.sum_residual = sum_residual
        self.geom_module_block = GeomModuleBlock(
                 fiber_in = fiber_in,
                 fiber_hidden = fiber_hidden,
                 fiber_out = fiber_out,
                 max_degree = max_degree,
                 num_heads  = num_heads,
                 channels_div = channels_div,
                 num_layers = 1,
                 fiber_edge = fiber_edge,
                 return_type = return_type,
                 pooling = pooling,
                 norm = norm,
                 use_layer_norm = use_layer_norm,
                 tensor_cores = tensor_cores,
                 low_memory = low_memory,
                 **kwargs
        )

        self.topo_module_block = TopoModuleBlock(
            topo_output_fiber = topo_output_fiber,
            embedding_dim = topo_embedding_dim,
            topo_input_fiber  = topo_input_fiber,
            topo_num_heads = topo_num_heads,
            dropout = topo_dropout,
            **kwargs)

        if self.comb_type == 'fctp':
            self.topo_irreps = fiber_to_irreps(topo_output_fiber)
            self.geom_irreps = fiber_to_irreps(fiber_out)
            self.comb_module = o3.FullyConnectedTensorProduct(irreps_in1 = self.topo_irreps,
                                         irreps_in2 = self.geom_irreps,
                                         irreps_out = self.geom_irreps,
                                         internal_weights = True)

        elif self.comb_type == 'attn':
            global_fiber = Fiber({'0': fiber_out[0]}) # Only zero order features!
            self.geom_fiber_to_k = LinearSE3(global_fiber, global_fiber)
            self.topo_fiber_to_q = LinearSE3(topo_output_fiber, global_fiber) # project topo into same fiber channels as fiber_out
            self.geom_fiber_to_v = LinearSE3(global_fiber, global_fiber)
            self.comb_norm = nn.LayerNorm(fiber_out[0])
        else:
            raise NotImplementedError
        
        self.weights = None

    def forward(self, node_feats: Dict[str, Tensor], 
                edge_feats: Optional[Dict[str, Tensor]],
                graph: DGLGraph,
                basis: Optional[Dict[str, Tensor]],
                topo_feats: Dict[str, Tensor]) -> Dict[str, Tensor]:
        #print(f'basis in contractrepsblock: {basis}')
        node_geom_reps = self.geom_module_block(graph, node_feats, edge_feats, basis, topo_feats)
        #print(f"Node geom reps variance: {torch.var(node_geom_reps['0'], axis = 0).squeeze()}")
        #print([x.shape for x in node_geom_reps.values()])
        topo_reps = self.topo_module_block(topo_feats) # type(topo_reps) = Dict[str: Tensor]
        #print(f"topo_reps: {topo_reps['0']}")
        #print(f"topo_reps_shape: {[x.shape for x in topo_reps.values()]}")
        #TODO: refactor these assert statements to use fiber dict instead!
        #assert topo_reps.shape[-1] == self.topo_irreps.dim, "Topo irreps have wrong tensor dim - input tensor should be (None, 300) for a 0ex300 irreps"
        #assert node_geom_reps.shape[-1] == self.geom_irreps.dim, 'Geom irreps have wrong tensor dim - input tensor should be (None, 288) for a 32x0e+32x1o+32x2e irreps'

        #Flatten node_reps, topo_reps into an e3nn style tensor - just concat all the degree reps
        #e3nn degree indexing goes 0, 1, 2, etc... - see printed rotation matrix at https://docs.e3nn.org/en/latest/guide/irreps.html
        # if self.comb_type == 'fctp':
        #     node_geom_reps = fiber_dict_to_flat(node_geom_reps)
        #     topo_reps = fiber_dict_to_flat(topo_reps)
        #     comb_irreps = self.comb_module.forward(topo_reps, node_geom_reps)
        #     comb_irreps= fiber_dict_from_flat(comb_irreps)
        #     return comb_irreps # Should be a (None, 288) tensor

        #elif self.comb_type == 'attn':
            # Cross attention on zero order features via learned linear projections of zero order features
        geom_k = self.geom_fiber_to_k(node_geom_reps) # {'0': ..., 32}
        topo_reps = self.broadcast_global(graph, topo_reps)
        topo_q = self.topo_fiber_to_q(topo_reps)
        geom_v = self.geom_fiber_to_v(node_geom_reps) 
        indices = torch.stack(graph.edges())
        N = graph.num_nodes()
        A = dglsp.spmatrix(indices, shape=(N, N))
        #print(f'node_geom_reps before  attn: {node_geom_reps}')
        #vals, self.weights = self.scaled_dot_product(topo_q['0'], geom_k['0'], geom_v['0'])
        vals, self.weights = self.graph_dot_product(A, topo_q['0'].squeeze(), geom_k['0'].squeeze(), geom_v['0'].squeeze())
        #print(f'after scp: vals.shape{vals.shape}, geom_v.shape{geom_v["0"].shape}')
        vals = geom_v['0'] + vals.unsqueeze(-1)
        vals = self.comb_norm(vals.squeeze())
        node_geom_reps['0'] = vals.unsqueeze(-1)
        #print(f'node_geom_reps update mag: {torch.sub(node_geom_reps["0"], vals)}')
        #node_geom_reps['0'] = vals
        if self.sum_residual:
            node_feats = aggregate_residual(node_feats, node_geom_reps, 'add')
        else:
            node_feats = node_geom_reps
        return node_feats

    def scaled_dot_product(self, q, k, v, mask=None):
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        attention = torch.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        return values, attention
    
    def graph_dot_product(self, A, q, k, v, mask = None):
        d_k = q.shape[-1]
        attn = dglsp.bsddmm(A, q, k.transpose(-2, -1))  # [N, N, nh]
        #print(f'attn shape: {attn.shape}')
        attn = attn / math.sqrt(d_k)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        attn = attn.softmax()
        at_wts = torch.sum(attn.val, dim = -1)
        #print(f'attention.val.shape {attn.val.shape}')
        out = dglsp.bspmm(attn, v)
        return out, at_wts
    
    def broadcast_global(self, graph: DGLGraph, global_feats: Dict[str, Tensor]):
        return {
            degree:  dgl.broadcast_nodes(graph, feats.squeeze()).unsqueeze(-1)
            for degree, feats in global_feats.items()
        }

class GeomModuleBlock(nn.Module):
    # Use convolution to update the equivariant repr.

    def __init__(self,
                 fiber_in: Fiber,
                 fiber_hidden: Fiber,
                 fiber_out: Fiber,
                 max_degree: int,
                 num_heads: int,
                 channels_div: int,
                 num_layers: int = 1,
                 fiber_edge: Fiber = Fiber({}),
                 return_type: Optional[int] = None,
                 pooling: Optional[Literal['avg', 'max']] = None,
                 norm: bool = True,
                 use_layer_norm: bool = True,
                 tensor_cores: bool = False,
                 low_memory: bool = True,
                 **kwargs):
        """
        :param num_layers:          Number of attention layers
        :param fiber_in:            Input fiber description
        :param fiber_hidden:        Hidden fiber description
        :param fiber_out:           Output fiber description
        :param fiber_edge:          Input edge fiber description
        :param num_heads:           Number of attention heads
        :param channels_div:        Channels division before feeding to attention layer
        :param return_type:         Return only features of this type
        :param pooling:             'avg' or 'max' graph pooling before MLP layers
        :param norm:                Apply a normalization layer after each attention block
        :param use_layer_norm:      Apply layer normalization between MLP layers
        :param tensor_cores:        True if using Tensor Cores (affects the use of fully fused convs, and padded bases)
        :param low_memory:          If True, will use slower ops that use less memory
        """
        super().__init__()
        self.num_layers = num_layers
        self.fiber_edge = fiber_edge
        self.num_heads = num_heads
        self.channels_div = channels_div
        self.return_type = return_type
        self.pooling = pooling
        self.max_degree = max_degree
        self.tensor_cores = tensor_cores
        self.low_memory = low_memory

        # if low_memory:
        #     self.fuse_level = ConvSE3FuseLevel.NONE
        # else:
        #     # Fully fused convolutions when using Tensor Cores (and not low memory mode)
        #     self.fuse_level = ConvSE3FuseLevel.FULL if tensor_cores else ConvSE3FuseLevel.PARTIAL

        self.fuse_level=ConvSE3FuseLevel.NONE
        graph_modules = []
        for i in range(num_layers):
            graph_modules.append(AttentionBlockSE3(fiber_in=fiber_in,
                                                   fiber_out=fiber_hidden,
                                                   fiber_edge=fiber_edge,
                                                   num_heads=num_heads,
                                                   channels_div=channels_div,
                                                   use_layer_norm=use_layer_norm,
                                                   max_degree=self.max_degree,
                                                   fuse_level=self.fuse_level,
                                                   low_memory=low_memory))
            if norm:
                graph_modules.append(NormSE3(fiber_hidden))
            fiber_in = fiber_hidden

        graph_modules.append(ConvSE3(fiber_in=fiber_in,
                                     fiber_out=fiber_out,
                                     fiber_edge=fiber_edge,
                                     self_interaction=True,
                                     use_layer_norm=use_layer_norm,
                                     max_degree=self.max_degree,
                                     fuse_level=self.fuse_level,
                                     low_memory=low_memory))
        self.graph_modules = Sequential(*graph_modules)

    def forward(self, graph: DGLGraph, node_feats: Dict[str, Tensor],
                edge_feats: Optional[Dict[str, Tensor]],
                basis: Optional[Dict[str, Tensor]],
                topo_feats: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # Compute bases in case they weren't precomputed as part of the data loading
        basis = basis or get_basis(graph.edata['rel_pos'], max_degree=self.max_degree, compute_gradients=False,
                                   use_pad_trick=self.tensor_cores and not self.low_memory,
                                   amp=torch.is_autocast_enabled())
        # basis = update_basis_with_fused(basis, self.max_degree, use_pad_trick=self.tensor_cores and not self.low_memory,
        #                         fully_fused=self.fuse_level == ConvSE3FuseLevel.FULL)

        node_feats = self.graph_modules(node_feats, edge_feats, graph=graph, basis=basis, topo_feats = topo_feats)

        # if self.return_type is not None:
        #     print('returning a tensor')
        #     return node_feats[str(self.return_type)]
        
        
        
        #TODO: Fix this hardcoding - for some reason, the graph modules here return a tensor and not a 
        # dict of {str: Tensor}

        return node_feats


class TopoModuleBlock(nn.Module):
    # Wrapper for BettiAttention - performs global attention on topological feature vectors.
    # Feature vectors are input as fibers of shape {0, input_feature_size} to account for rotatinal invariance.
    def __init__(self,
            topo_output_fiber: Fiber,
            embedding_dim: int,
            topo_input_fiber: Fiber,
            topo_num_heads: int,
            dropout = 0.0,
            **kwargs):
        super().__init__()
        #self.fiber_in = input_fiber # Should be Fiber({'0': input_size})
        self.betti_block = BettiAttention(output_feature_dim = topo_output_fiber[0], # NO STRINGS FOR INDEXING!! The dimensions are integers!
                                          embedding_dim = embedding_dim, 
                                          input_size = topo_input_fiber[0],
                                          num_encoder_layers = 1,
                                          topo_num_heads = topo_num_heads,
                                          dropout = dropout, **kwargs)
    def forward(self, topo_vector_in: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = self.betti_block(topo_vector_in['0'])  # (batch, 3, output_feature_dim)
        x = x.mean(dim=1, keepdim=True)            # (batch, 1, output_feature_dim) — pool over betti tokens
        return {'0': x}