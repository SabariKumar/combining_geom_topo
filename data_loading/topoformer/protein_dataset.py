import os
import glob
from typing import Union, Tuple
import click
import dgl # type: ignore
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys


from Bio.PDB.DSSP import DSSP
from Bio.PDB import PDBParser
from Bio.Data.IUPACData import protein_letters_1to3

from graphein.protein.graphs import read_pdb_to_dataframe, process_dataframe, deprotonate_structure, convert_structure_to_centroids, subset_structure_to_atom_type, filter_hetatms, remove_insertions # type: ignore
from graphein.protein.graphs import initialise_graph_with_metadata, add_nodes_to_graph, compute_edges # type: ignore
#from graphein.protein.graphs import annotate_node_metadata, annotate_graph_metadata, annotate_edge_metadata
from graphein.protein.edges.distance import (add_peptide_bonds, # type: ignore
                                             add_hydrophobic_interactions,
                                             add_hydrogen_bond_interactions,
                                             add_disulfide_interactions,
                                             add_ionic_interactions,
                                             add_aromatic_interactions,
                                             add_aromatic_sulphur_interactions,
                                             add_cation_pi_interactions
                                            )
from graphein.protein.features.nodes.amino_acid import amino_acid_one_hot # type: ignore
from graphein.utils.utils import onek_encoding_unk # type: ignore
from graphein.protein.resi_atoms import STANDARD_AMINO_ACIDS # type: ignore

#Suppress annoying graphein logs
from loguru import logger as log # type: ignore
log.remove()
log.add(sys.stdout, level="ERROR") 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

torch.multiprocessing.set_sharing_strategy('file_system') 
def add_dssp_rsas(G:nx.Graph) -> nx.Graph:
    # Manual implementation of dssp computed rsa for graphein's API. Bio's dssp_dict_from_pdb doesn't work for some reason,
    # so need to do it this way.
    # Deprecated 02/14/2024 for get_node_features
    # TODO: fix dssp_dict_from_pdb with graphein's DSSPConfig wrapper. Everything's set properly, don't
    # know why this keeps silently failing.  
    p = PDBParser()
    model = p.get_structure('test', G.graph['path'])[0]
    dssp = DSSP(model, G.graph['path'])
    rsas = [dssp[x][3] for x in list(dssp.keys())] # 3 corresponds to the Relative ASA index in the DSSP tuple
    nx.set_node_attributes(G, dict(zip(G.nodes, rsas)), 'rsa')
    return G

def one_hot_edges(u, v, d) -> None:
    #Adds a one hot encoding corresponding to the built-in interaction types in graphein 
    b_type = list(d['kind'])[0]
    edge_enc = {'peptide_bond': 0,
                'hydrophobic': 1,
                'hbond': 2,
                'disulfide': 3,
                'ionic': 4,      
                'aromatic': 5,
                'aromatic-sulphur': 6,
                'cation-pi': 7}
    enc = np.zeros(len(edge_enc.keys()))
    enc[edge_enc[b_type]] = 1
    d['edge_one_hot'] = enc
    return(enc)

def clean_pdb(inpfile) -> Tuple[str, str]:
    #Adds dummy pdb header info so DSSP doesn't complain. See https://github.com/PDB-REDO/dssp/issues/1#issuecomment-1778532504
    basename = os.path.splitext(inpfile)[0]
    with open(inpfile, 'r') as infile:
        contents = infile.readlines()
        outfile_name = basename + '_dssp_edits.pdb'
        with open(outfile_name, 'w') as outfile:
            contents.insert(1, 'MODEL     1\n')
            contents.insert(1, 'CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1          \n')
            contents.insert(-1, 'ENDMDL\n')
            for line in contents[1:]:
                if 'PARENT' not in line:
                    outfile.write(line)
    return outfile_name, basename

def process_dssp_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes a DSSP DataFrame to make indexes align with node IDs

    :param df: pd.DataFrame containing the parsed output from DSSP.
    :type df: pd.DataFrame
    :return: pd.DataFrame with node IDs
    :rtype: pd.DataFrame
    """
    # Convert 1 letter aa code to 3 letter
    amino_acids = df["aa"].tolist()
    for i, amino_acid in enumerate(amino_acids):
        amino_acids[i] = protein_letters_1to3[amino_acid].upper()
    df["aa"] = amino_acids
    # Construct node IDs
    node_ids = []
    for i, row in df.iterrows():
        node_id = row["chain"] + ":" + row["aa"] + ":" + str(row["resnum"])
        node_ids.append(node_id)
    df["node_id"] = node_ids
    df.set_index("node_id", inplace=True)
    return df

class ProteinDataset(Dataset):

    num_bonds = 8
    node_feature_size = 25
    def __init__(self, pdb_dir = '/home/sabari/ProteinSol/combining_geom_topo/data/train',
                sol_df = '/home/sabari/ProteinSol/combining_geom_topo/data/csvs/training_set.csv',
                mode: str = 'train', use_barcodes = False,
                processed_dir = '/home/sabari/ProteinSol/combining_geom_topo/data/train',
                barcode_dir = '/home/sabari/ProteinSol/combining_geom_topo/data/train',
                esm_dir = '/home/sabari/ProteinSol/combining_geom_topo/data/train',
                force_rebuild = False):
        self.pdb_dir = pdb_dir
        self.sol_df = sol_df
        self.mode = mode
        self.use_barcodes = use_barcodes
        self.processed_dir = processed_dir
        self.barcode_dir = barcode_dir
        self.esm_dir = esm_dir
        self.force_rebuild = force_rebuild
        self.sol_data = self._load_sol_labels()
        self.data_list = self._get_data_list()

    def __len__(self):
        return len(self.data_list)
    
    def _load_sol_labels(self):
        self.sol_df = pd.read_csv(self.sol_df)
        sol_data = self.sol_df.set_index('sid')
        return sol_data
    
    def _get_data_list(self):
        return sorted(glob.glob(os.path.join(self.pdb_dir, '*.pdb')))
        
    def _build_data(self, input_pdb):
        if '_dssp_edits' in input_pdb:
            pdb_path = input_pdb
        else:
            pdb_path, _ = clean_pdb(input_pdb)
        base = Path(pdb_path).stem.split('_')[0]
        esm_emb = torch.load(str(os.path.join(self.esm_dir, str(base) + '.pt'))).cpu().numpy()
        #Build the protein graph using graphein's low level API
        pdb_processing_funcs = [deprotonate_structure, 
                                convert_structure_to_centroids, 
                                remove_insertions]
        raw_df = read_pdb_to_dataframe(path = pdb_path)
        df = process_dataframe(raw_df, atom_df_processing_funcs = pdb_processing_funcs)
        g = initialise_graph_with_metadata(protein_df=df,
                                    raw_pdb_df=raw_df, 
                                    path = pdb_path, # Store this for traceability
                                    granularity = "centroid")
        f = add_nodes_to_graph(g)
        g = compute_edges(g, get_contacts_config=None, funcs=[add_peptide_bonds,
                                                add_hydrophobic_interactions,
                                                add_hydrogen_bond_interactions,
                                                add_disulfide_interactions,
                                                add_ionic_interactions,
                                                add_aromatic_interactions,
                                                add_aromatic_sulphur_interactions,
                                                add_cation_pi_interactions])
        dat = dgl.from_networkx(g)
        dat.ndata['pos'] = torch.tensor(self._get_node_positions(g))
        dat.ndata['attr'] = torch.tensor(self._get_node_features(g, esm_emb, base))
        dat.edata['edge_attr'] = torch.tensor(self._get_edge_features(g))
        dat.edata['rel_pos'] = torch.tensor(self._get_edge_lengths(g))
        dat = dgl.add_self_loop(dat)
        dat = dgl.to_float(dat)
        sol = int(self.sol_data.loc[int(base), "solubility"])
        target = torch.tensor(sol).type(torch.FloatTensor)

        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save(dat, os.path.join(self.processed_dir, base + '_input.pt'))
        torch.save(target, os.path.join(self.processed_dir, base + '_target.pt'))
        return dat, target
    
    def __getitem__(self, idx):
        pdb_path = self.data_list[idx]
        base = Path(pdb_path).stem.split('_')[0]

        input_path = os.path.join(self.processed_dir, base + '_input.pt')
        target_path = os.path.join(self.processed_dir, base + '_target.pt')

        if not self.force_rebuild and os.path.exists(input_path) and os.path.exists(target_path):
            dat = torch.load(input_path)
            target = torch.load(target_path)
        else:
            dat, target = self._build_data(pdb_path)

        if self.use_barcodes:
            barcode_list = sorted(glob.glob(os.path.join(self.barcode_dir, f'{base}_*_emb_b*.pt')))
            if len(barcode_list) != 3:
                # Fall back to legacy naming pattern
                barcode_list = sorted(glob.glob(os.path.join(self.barcode_dir, f'*_{base}_emb*.pt')))
            b0 = torch.load(barcode_list[0])
            b1 = torch.load(barcode_list[1])
            b2 = torch.load(barcode_list[2])
            barcode = torch.cat([b0, b1, b2], -1)
        else:
            barcode = None

        return dat, barcode, target, base

    def _get_node_positions(self, G: nx.graph) -> np.array:
        node_pos = []
        for _, d in G.nodes(data = True):
            node_pos.append(d['coords'])
        node_pos = np.array(node_pos)
        return node_pos
    
    def _get_node_features(self, G: nx.Graph, esm_emb:np.array, base) -> np.array:
        # Graphein's add_node_attributes isn't setting DGL node attibutes properly, workaround.
        # Returns a [num_aas, 25, 1] np array, set as 'f' to use the default SE(3) xformer keys (matching DGL's QM9)
        # TODO: fix dssp_dict_from_pdb with graphein's DSSPConfig wrapper. Everything's set properly, don't
        # know why this keeps silently failing.  
        p = PDBParser()
        model = p.get_structure('test', G.graph['path'])[0]
        try:
            dssp = DSSP(model, G.graph['path'])
            rsas = np.array([dssp[x][3] for x in list(dssp.keys())]).reshape(-1, 1)
            aas = [dssp[x][1] for x in list(dssp.keys())]
            aas_ohe = np.array([onek_encoding_unk(x, STANDARD_AMINO_ACIDS) for x in aas]).astype(float)
            return np.concatenate([aas_ohe,rsas, esm_emb], -1).reshape(-1, 1280+25, 1)
        except Exception as e:
            print(f'Error generating features for sid {base}: {e}')
            return np.zeros((G.number_of_nodes(), 1280+25, 1))
    
    def _get_edge_lengths(self, G: nx.Graph) -> np.array:
        node_dict = dict(G.nodes(data=True))
        edge_vec = []
        for u, v, _ in G.edges(data=True):
            edge_vec.append(node_dict[u]['coords'] - node_dict[v]['coords'])
        edge_vec = np.array(edge_vec)
        # Accounting for back-edges
        edge_vec = np.concatenate([edge_vec, edge_vec*-1])
        return edge_vec
    
    def _get_edge_features(self, G: nx.Graph) -> np.array:
        edge_feat = []
        for u, v, d in G.edges(data = True):
            edge_feat.append(one_hot_edges(u, v, d))
        edge_feat = np.array(edge_feat).astype(float)
        edge_feat = np.concatenate([edge_feat, edge_feat])
        return edge_feat


def collate_barcodes(samples):
    graphs, barcodes, target, base = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, barcodes, target, base

@click.command()
@click.option('--pdb_dir', default = '/home/sabari/ProteinSol/combining_geom_topo/data/train')
@click.option('--sol_df', default = '/home/sabari/ProteinSol/combining_geom_topo/data/csvs/training_set.csv')
@click.option('--mode', default = 'train')
@click.option('--use_barcodes', default = False)
@click.option('--processed_dir', default = '/home/sabari/ProteinSol/combining_geom_topo/data/train')
@click.option('--barcode_dir', default = '/home/sabari/ProteinSol/combining_geom_topo/data/train')
@click.option('--esm_dir', default = '/home/sabari/ProteinSol/combining_geom_topo/data/train')
@click.option('--force_rebuild', is_flag = True, default = False, help = 'Rebuild all cached .pt files even if they exist')
@click.option('--batch_size', default = 32)
@click.option('--shuffle', default = True)
@click.option('--num_workers', default = 16)
def build_dataset(pdb_dir,
                  sol_df,
                  mode,
                  use_barcodes,
                  processed_dir,
                  barcode_dir,
                  esm_dir,
                  force_rebuild,
                  batch_size,
                  shuffle,
                  num_workers):
    dataset = ProteinDataset(pdb_dir = pdb_dir,
                             sol_df = sol_df,
                             mode = mode,
                             use_barcodes = use_barcodes,
                             processed_dir = processed_dir,
                             barcode_dir = barcode_dir,
                             esm_dir = esm_dir,
                             force_rebuild = force_rebuild)
    dataloader = DataLoader(dataset,
                            batch_size = batch_size,
                            shuffle = shuffle,
                            collate_fn = collate_barcodes,
                            num_workers = num_workers)
    for _ in dataloader:
        pass

if __name__ == '__main__':
    build_dataset()