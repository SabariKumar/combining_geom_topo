import logging
from typing import Optional, Literal, Dict

import torch
import torch.nn as nn
import dgl # type: ignore
from dgl import DGLGraph # type: ignore
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
from model.topoformer.utils import str2bool, aggregate_residual
from model.topoformer.fiber import Fiber, fiber_to_irreps, fiber_dict_to_flat, fiber_dict_from_flat
from model.topoformer.layers.bettiattention import BettiAttention

class Sequential(nn.Sequential):
    """ Sequential module with arbitrary forward args and kwargs. Used to pass graph, basis and edge features. """

    def forward(self, input, *args, **kwargs):
        for module in self:
            #print(f'Working on module: {module}')
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

class TopoformerPooled(nn.Module):
    def __init__(self,
                 fiber_in: Fiber,
                 fiber_out: Fiber,
                 fiber_edge: Fiber,
                 num_degrees: int,
                 num_channels: int,
                 output_dim: int,
                 **kwargs):
        super().__init__()
        kwargs['pooling'] = kwargs['pooling'] or 'avg'
        self.transformer = Topoformer(
            fiber_in = fiber_in,
            fiber_hidden = Fiber.create(num_degrees, num_channels),
            fiber_out = fiber_out,
            fiber_edge = fiber_edge,
            return_type = 0,
            **kwargs
        )
        n_out_features = fiber_out.num_features
        dropout = kwargs.get('dropout', 0.0)
        output_layer = nn.Linear(n_out_features, output_dim)
        nn.init.zeros_(output_layer.weight)
        nn.init.constant_(output_layer.bias, 0.0)
        self.mlp = nn.Sequential(
            nn.Linear(n_out_features, n_out_features),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            output_layer
        )

    def forward(self, graph, node_feats, topo_feats, edge_feats, basis=None):
        feats = self.transformer(graph, node_feats, topo_feats, edge_feats, basis).squeeze(-1)
        y = self.mlp(feats).squeeze(-1)
        return y

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("Model architecture")
        Topoformer.add_argparse_args(parser)
        parser.add_argument('--num_degrees',
                            help='Number of degrees to use. Hidden features will have types [0, ..., num_degrees - 1]',
                            type=int, default=4)
        parser.add_argument('--num_channels', help='Number of channels for the hidden features', type=int, default=32)
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
                fuse_level: ConvSE3FuseLevel = ConvSE3FuseLevel.NONE,
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
        self.fuse_level = fuse_level

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
            fuse_level = self.fuse_level,
            comb_type = comb_type, #fctp or conv
            topo_output_fiber = topo_output_fiber,
            topo_embedding_dim = topo_embedding_dim,
            topo_input_fiber = topo_input_fiber,
            topo_num_heads = topo_num_heads,
            topo_dropout = topo_dropout,
        ))

        for _ in range(num_layers - 1):
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
                fuse_level = self.fuse_level,
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
        if pooling is not None:
            assert return_type is not None, 'return_type must be specified when pooling'
            self.pooling_module = GPooling(pool=pooling, feat_type=return_type)

    def forward(self, graph: DGLGraph, node_feats: Dict[str, Tensor], topo_feats: Dict[str, Tensor],
                edge_feats: Optional[Dict[str, Tensor]] = None,
                basis: Optional[Dict[str, Tensor]] = None):
        
        if self.use_topo_projection:
            topo_feats['0'] = self.topo_mlp(topo_feats['0'])

        node_feats = self.contract_modules(node_feats,
                                           edge_feats,
                                           graph = graph,
                                           basis = basis,
                                           topo_feats = topo_feats)
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
        parser.add_argument('--eq_dropout', type=float, default=0.0,
                            help='Dropout probability applied to equivariant features between ContractRepsBlocks')

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
                 fuse_level: ConvSE3FuseLevel = ConvSE3FuseLevel.NONE,
                 comb_type: str = 'attn', #fctp or attn
                 topo_output_fiber: Fiber = Fiber({'0': 300}),
                 topo_embedding_dim: int = 300,
                 topo_input_fiber: Fiber = Fiber({'0': 300}),
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
                 num_layers = 2,
                 fiber_edge = fiber_edge,
                 return_type = return_type,
                 pooling = pooling,
                 norm = norm,
                 use_layer_norm = use_layer_norm,
                 tensor_cores = tensor_cores,
                 low_memory = low_memory,
                 fuse_level = fuse_level
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
            self.topo_fiber_to_q = LinearSE3(Fiber({'0': 300}), global_fiber) # project topo into same fiber channels as fiber_out
            self.geom_fiber_to_v = LinearSE3(global_fiber, global_fiber)
            self.comb_norm = nn.LayerNorm(fiber_out[0])
        else:
            raise NotImplementedError
        


    def forward(self, node_feats: Dict[str, Tensor], 
                edge_feats: Optional[Dict[str, Tensor]],
                graph: DGLGraph,
                basis: Optional[Dict[str, Tensor]],
                topo_feats: Dict[str, Tensor]) -> Dict[str, Tensor]:
        
        node_geom_reps = self.geom_module_block(graph, node_feats, edge_feats, basis, topo_feats)
        topo_reps = self.topo_module_block(topo_feats) # type(topo_reps) = Dict[str: Tensor]
        geom_k = self.geom_fiber_to_k(node_geom_reps) # {'0': ..., 32}
        topo_reps = self.broadcast_global(graph, topo_reps)
        topo_q = self.topo_fiber_to_q(topo_reps)
        geom_v = self.geom_fiber_to_v(node_geom_reps) 
        vals, self.weights = self.scaled_dot_product(topo_q['0'], geom_k['0'], geom_v['0'])
        vals = geom_v['0'] + vals
        vals = self.comb_norm(vals.squeeze())
        node_geom_reps['0'] = vals.unsqueeze(-1)
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
                 fuse_level: ConvSE3FuseLevel = ConvSE3FuseLevel.NONE,
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
        self.fuse_level = fuse_level
        graph_modules = []
        # for i in range(num_layers):
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
        basis = basis or get_basis(graph.edata['rel_pos'], max_degree=self.max_degree, compute_gradients=False,
                                   use_pad_trick=self.tensor_cores and not self.low_memory,
                                   amp=torch.is_autocast_enabled())
        
        basis = update_basis_with_fused(basis, self.max_degree, use_pad_trick=self.tensor_cores and not self.low_memory,
                        fully_fused=self.fuse_level == ConvSE3FuseLevel.FULL)
        
        node_feats = self.graph_modules(node_feats, edge_feats, graph=graph, basis=basis, topo_feats = topo_feats)
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