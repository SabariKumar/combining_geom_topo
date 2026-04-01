from tqdm import tqdm
import dgl # type: ignore
import pathlib
import numpy as np
import os
import math
import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from abc import ABC
from torch.utils.data import DataLoader, DistributedSampler, Dataset, random_split, Subset

import sys
sys.path.append("/home/sabari/ProteinSol/topoformer")
from data_loading.topoformer.data_module import DataModule
from model.topoformer.basis import get_basis
from model.topoformer.utils import get_local_rank, str2bool, using_tensor_cores
from data_loading.topoformer.protein_dataset import ProteinDataset

from sklearn.model_selection import KFold
RANDOM_STATE = 42

def _get_relative_pos(graph):
    x = graph.ndata['pos']
    src, dst =graph.edges()
    rel_pos = x[dst] - x[src]
    return rel_pos

def _get_split_sizes(full_dataset: Dataset):
    len_full = len(full_dataset)
    len_train = int(0.9 * len_full)
    len_val = int(0.1 * len_full)
    len_test = len_full - len_train - len_val
    return len_train, len_val, len_test

def _get_split_sizes_external(full_dataset: Dataset):
    len_full = len(full_dataset)
    #len_train = int(0.9 * len_full)
    #len_val = int(0.1 * len_full)
    len_val = 1
    #len_test = len_full - len_train - len_val
    print(f'Length of full dataset: {len(full_dataset)}')
    return len_full-len_val-1, len_val, 1

class CachedBasesProteinDataset(ProteinDataset):
    """ Dataset extending the QM9 dataset from DGL with precomputed (cached in RAM) pairwise bases """

    def __init__(self, bases_kwargs: dict, batch_size: int, num_workers: int, *args, **kwargs):
        """
        :param bases_kwargs:  Arguments to feed the bases computation function
        :param batch_size:    Batch size to use when iterating over the dataset for computing bases
        """
        super().__init__(*args, **kwargs)

        self.bases_kwargs = bases_kwargs
        self.batch_size = batch_size
        self.bases = None
        self.sids = None
        self.num_workers = num_workers
        self.ne_cumsum = self._calc_cum_sum()
        self.load()
        self.check_basis()

    def _calc_cum_sum(self):
        # Need this to get the edge cumsum value for basis calc indexing
        # Can't put this in load since dataloader iteration needs to be fully sequential!
        dataloader = DataLoader(self, shuffle=False, batch_size=1, num_workers=1,
                                collate_fn=lambda samples: dgl.batch([sample[0] for sample in samples]))
        n_edge = []
        sids = []
        for i, graph in tqdm(enumerate(dataloader), total=len(dataloader), desc='Computing aggregate edge numbers',
                            disable=get_local_rank() != 0):
            n_edge.append(graph.number_of_edges())
        n_edge = np.array(n_edge)
        cum_sum = np.insert(np.cumsum(n_edge), 0, [0])
        return(cum_sum)
    
    def load(self):
        # Iterate through the dataset and compute bases (pairwise only)
        # Potential improvement: use multi-GPU and gather
        dataloader = DataLoader(self, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers,
                                collate_fn=lambda samples: (dgl.batch([sample[0] for sample in samples]), [sample[-1] for sample in samples]))
        bases = []
        sids = []
        for i, graph in tqdm(enumerate(dataloader), total=len(dataloader), desc='Precomputing QM9 bases',
                             disable=get_local_rank() != 0):
            rel_pos = _get_relative_pos(graph[0])
            # Compute the bases with the GPU but convert the result to CPU to store in RAM
            bases.append({k: v.cpu() for k, v in get_basis(rel_pos.cuda(), **self.bases_kwargs).items()})
            sids.append(graph[-1])
        #print(f'bases after precompute loop: {bases}')
        self.bases = bases  # Assign at the end so that __getitem__ isn't confused
        self.sids = sids

    def __getitem__(self, idx: int):
        graph, barcode, label, sid = super().__getitem__(idx)
        graph = dgl.to_float(graph)
        #print(f'bases in get_item: {self.bases}')
        if self.bases:
            # Break out the bases per batch - no need to include those we don't need. Use the cum. sum. to keep track
            # of accumuluation
            bases_idx = idx // self.batch_size
            bases_cumsum_idx = self.ne_cumsum[idx] - self.ne_cumsum[bases_idx * self.batch_size]
            bases_cumsum_next_idx = self.ne_cumsum[idx + 1] - self.ne_cumsum[bases_idx * self.batch_size]

            if self.use_barcodes:
                return graph, barcode, label, sid, {key: basis[bases_cumsum_idx:bases_cumsum_next_idx] for key, basis in
                                    self.bases[bases_idx].items()}
            else:
                return graph, label, sid, {key: basis[bases_cumsum_idx:bases_cumsum_next_idx] for key, basis in
                                    self.bases[bases_idx].items()}
        else:
            if self.use_barcodes:
                return graph, barcode, label, sid
            else:
                return graph, label, sid
            
    def check_basis(self):
        for base, sid in zip(self.bases, self.sids):
            if '0,0' not in base.keys():
                print(sid)
            
class ProteinDataModule(DataModule):
    """
    Datamodule wrapping https://docs.dgl.ai/en/latest/api/python/dgl.data.html#qm9edge-dataset
    Training set is 100k molecules. Test set is 10% of the dataset. Validation set is the rest.
    This includes all the molecules from QM9 except the ones that are uncharacterized.
    """

    # NODE_FEATURE_DIM = 25
    NODE_FEATURE_DIM = 25+1280
    EDGE_FEATURE_DIM = 8

    def __init__(self,
                 pdb_dir: pathlib.Path,
                 sol_df: pathlib.Path,
                 processed_dir: pathlib.Path,
                 use_barcodes: bool = False,
                 force_rebuild: bool = False,
                 barcode_dir = '../../data/soluprotgeom/processed/train/gudhi_embeddings',
                 task: str = 'homo',
                 batch_size: int = 240,
                 num_workers: int = 4,
                 num_degrees: int = 4,
                 amp: bool = False,
                 precompute_bases: bool = False,
                 external_test = None,
                 external_df = None,
                 external_barcode_dir = None,
                 stratified_split = None,
                 #external_test = '/home/sabari/ProteinSol/se3_transformer_nvidia/SE3Transformer/se3_transformer/data_loading/data/test_set',
                 #external_df = '/home/sabari/ProteinSol/se3_transformer_nvidia/SE3Transformer/se3_transformer/data_loading/data/test_set/test_set.csv',
                 #external_barcode_dir =  '/home/sabari/ProteinSol/se3_transformer_nvidia/SE3Transformer/se3_transformer/data_loading/data/test_set/barcodes',
                 **kwargs):
        self.pdb_dir = pdb_dir  # This, along with all the protein dataset stuff, needs to be before __init__ so that prepare_data has access to it
        self.sol_df = sol_df
        self.force_rebuild = force_rebuild
        self.processed_dir = processed_dir
        super().__init__(batch_size=batch_size, num_workers=num_workers, collate_fn=self._collate)
        self.amp = amp
        self.task = task
        self.batch_size = batch_size
        self.num_degrees = num_degrees
        print(f'precompute_bases: {precompute_bases}')
        if precompute_bases:
            bases_kwargs = dict(max_degree=num_degrees - 1, use_pad_trick=using_tensor_cores(amp), amp=amp)
            full_dataset = CachedBasesProteinDataset(bases_kwargs=bases_kwargs, batch_size=batch_size,
                                                     num_workers=num_workers,
                                                     pdb_dir = self.pdb_dir,
                                                     sol_df = self.sol_df,
                                                     mode = 'train',
                                                     use_barcodes = use_barcodes,
                                                     force_rebuild = self.force_rebuild,
                                                     processed_dir = self.processed_dir,
                                                     barcode_dir = barcode_dir)
            if external_test:
                test_dataset = CachedBasesProteinDataset(bases_kwargs=bases_kwargs, batch_size=batch_size,
                                                     num_workers=num_workers,
                                                     pdb_dir = external_test,
                                                     sol_df = external_df,
                                                     mode = 'test',
                                                     use_barcodes = use_barcodes,
                                                     force_rebuild = True,
                                                     barcode_dir = external_barcode_dir)
        else:
            full_dataset = ProteinDataset(pdb_dir = self.pdb_dir,
                              sol_df = self.sol_df,
                              mode = 'train',
                              use_barcodes = use_barcodes,
                              force_rebuild = self.force_rebuild,
                              processed_dir = self.processed_dir,
                              barcode_dir = barcode_dir)
            
        if stratified_split:
            train_ind, test_ind = get_strat_splits() #TODO
        else:
            # self.ds_train, self.ds_val, self.ds_test = random_split(full_dataset, _get_split_sizes(full_dataset),
            #                                                         generator=torch.Generator().manual_seed(0))
            self.ds_train, self.ds_val, self.ds_test = random_split(full_dataset, _get_split_sizes_external(full_dataset),
                                                        generator=torch.Generator().manual_seed(0))
        if external_test:
            self.ds_train = full_dataset
            self.ds_test = test_dataset

        train_targets = full_dataset.sol_df['solubility'][self.ds_train.indices].to_numpy(dtype = float)
        self.targets_mean = train_targets.mean()
        self.targets_std = train_targets.std()

    def prepare_data(self):
        pass  # Cache-through in __getitem__ handles pre-generation automatically

    def _collate(self, samples):
        graphs, barcodes, y, sids, *bases = map(list, zip(*samples))
        #print(f'collate bases{bases}')
        batched_graph = dgl.batch(graphs)
        edge_feats = {'0': batched_graph.edata['edge_attr'][:, :self.EDGE_FEATURE_DIM, None]}
        batched_graph.edata['rel_pos'] = _get_relative_pos(batched_graph)
        # get node features
        node_feats = {'0': torch.flatten(batched_graph.ndata['attr'][:, :self.NODE_FEATURE_DIM, None], start_dim = -2, end_dim = -1)}
        #node_feats = {'0': batched_graph.ndata['attr'][:, :self.NODE_FEATURE_DIM, None]}

        #print(f'node_feats in pdm collate: {node_feats["0"].shape}')
        #print(f'node_feats in pdm collate: {batched_graph.ndata["attr"][:, :self.NODE_FEATURE_DIM, None].shape}')

        barcode_feats = {'0': torch.stack(barcodes, dim = 0).float()}
        #targets = torch.cat(torch.tensor(y)) # Dont rescale when your labels are 0, 1! Rescaling changes labels to -1, 1
        targets = torch.tensor(y)
        #targets = (torch.cat(y) - self.targets_mean) / self.targets_std
        if bases:
            # collate bases
            all_bases = {
                key: torch.cat([b[key] for b in bases[0]], dim=0)
                for key in bases[0][0].keys()
            }
            #return batched_graph, barcodes, node_feats, edge_feats, all_bases, targets
            return sids, batched_graph, node_feats, barcode_feats, edge_feats, all_bases, targets
        else:
            #return batched_graph, barcodes, node_feats, edge_feats, targets
            return sids, batched_graph, node_feats, barcode_feats, edge_feats, targets
        
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("protein dataset")
        parser.add_argument('--precompute_bases', type=str2bool, nargs='?', const=True, default=False,
                            help='Precompute bases at the beginning of the script during dataset initialization,'
                                 ' instead of computing them at the beginning of each forward pass.')
        return parent_parser
    

class Protein5FoldsModule(DataModule):
    """
    Datamodule wrapping https://docs.dgl.ai/en/latest/api/python/dgl.data.html#qm9edge-dataset
    Training set is 100k molecules. Test set is 10% of the dataset. Validation set is the rest.
    This includes all the molecules from QM9 except the ones that are uncharacterized.
    """

    NODE_FEATURE_DIM = 25
    EDGE_FEATURE_DIM = 8

    def __init__(self,
                 n_fold: int,
                 pdb_dir: pathlib.Path,
                 sol_df: pathlib.Path,
                 processed_dir: pathlib.Path,
                 use_barcodes: bool = False,
                 force_rebuild: bool = False,
                 barcode_dir = '/home/sabari/ProteinSol/se3_transformer_nvidia/SE3Transformer/se3_transformer/data_loading/data/topological_embeddings/atol/',
                 task: str = 'homo',
                 batch_size: int = 240,
                 num_workers: int = 4,
                 num_degrees: int = 4,
                 amp: bool = False,
                 precompute_bases: bool = False,
                 **kwargs):
        self.pdb_dir = pdb_dir  # This, along with all the protein dataset stuff, needs to be before __init__ so that prepare_data has access to it
        self.sol_df = sol_df
        self.force_rebuild = force_rebuild
        self.processed_dir = processed_dir
        self.use_barcodes = use_barcodes
        super().__init__(batch_size=batch_size, num_workers=num_workers, collate_fn=self._collate)
        self.amp = amp
        self.task = task
        self.batch_size = batch_size
        self.num_degrees = num_degrees

        if precompute_bases:
            bases_kwargs = dict(max_degree=num_degrees - 1, use_pad_trick=using_tensor_cores(amp), amp=amp)
            full_dataset = CachedBasesProteinDataset(bases_kwargs=bases_kwargs, batch_size=batch_size,
                                                     num_workers=num_workers,
                                                     pdb_dir = self.pdb_dir,
                                                     sol_df = self.sol_df,
                                                     mode = 'train',
                                                     use_barcodes = self.use_barcodes,
                                                     force_rebuild = self.force_rebuild,
                                                     processed_dir = self.processed_dir,
                                                     barcode_dir = barcode_dir)
        else:
            full_dataset = ProteinDataset(pdb_dir = self.pdb_dir,
                              sol_df = self.sol_df,
                              mode = 'train',
                              use_barcodes = self.use_barcodes,
                              force_rebuild = self.force_rebuild,
                              processed_dir = self.processed_dir)
            
        kf = KFold(n_splits = 5, random_state = RANDOM_STATE, shuffle=True)    
        train_idx, test_idx = kf.split(full_dataset)[n_fold]
        self.ds_train = Subset(full_dataset, train_idx)
        self.ds_test = Subset(full_dataset, test_idx)
        self.ds_valid = Subset(full_dataset, test_idx)

        train_targets = full_dataset.sol_df['solubility'][self.ds_train.indices].to_numpy(dtype = float)
        self.targets_mean = train_targets.mean()
        self.targets_std = train_targets.std()

    def prepare_data(self):
        pass  # Cache-through in __getitem__ handles pre-generation automatically

    def _collate(self, samples):
        graphs, barcodes, y, sids, *bases = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        edge_feats = {'0': batched_graph.edata['edge_attr'][:, :self.EDGE_FEATURE_DIM, None]}
        batched_graph.edata['rel_pos'] = _get_relative_pos(batched_graph)
        # get node features
        node_feats = {'0': torch.flatten(batched_graph.ndata['attr'][:, :self.NODE_FEATURE_DIM, None], start_dim = -2, end_dim = -1)}
        if self.use_barcodes:
            barcode_feats = {'0': torch.stack(barcodes, dim = 0).float()}
        else:
            barcode_feats = {'0': None}
        targets = torch.cat(y) # Dont rescale when your labels are 0, 1! Rescaling changes labels to -1, 1
        #targets = (torch.cat(y) - self.targets_mean) / self.targets_std

        if bases:
            # collate bases
            all_bases = {
                key: torch.cat([b[key] for b in bases[0]], dim=0)
                for key in bases[0][0].keys()
            }
            #return batched_graph, barcodes, node_feats, edge_feats, all_bases, targets
            return sids, batched_graph, node_feats, barcode_feats, edge_feats, all_bases, targets
        else:
            #return batched_graph, barcodes, node_feats, edge_feats, targets
            return sids, batched_graph, node_feats, barcode_feats, edge_feats, targets
