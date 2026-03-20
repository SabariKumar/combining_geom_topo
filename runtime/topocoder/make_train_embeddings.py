"""
Generate Topocoder embeddings for all training PDBs in data/train/.

Two-step pipeline:
1. Extract CA coordinates from PDBs (one per unique SID)
2. Run trained Topocoder inference to produce embeddings for Betti 0, 1, 2

Usage:
    python make_train_embeddings.py \
        --pdb_dir /home/sabari/ProteinSol/combining_geom_topo/data/train \
        --trained_model_dir /home/sabari/ProteinSol/combining_geom_topo/results/topocoder_models
"""

import os
import glob
from typing import Union

import click
import numpy as np
import numpy.typing as npt
import torch
import torch.utils.data as data
from Bio.PDB import PDBParser
from tqdm import tqdm

import sys
sys.path.append("/home/sabari/ProteinSol/combining_geom_topo")

from model.topocoder.topocoder import TopoCoder


def get_CA_coords(input_path: Union[str, bytes, os.PathLike]) -> npt.NDArray[np.float32]:
    p = PDBParser(QUIET=True)
    structure = p.get_structure("input", input_path)
    for model in structure:
        for chain in model:
            CA_coords = [residue["CA"].get_coord() for residue in chain if "CA" in residue]
    return np.array(CA_coords, dtype="float32")


def get_unique_pdbs(pdb_dir):
    """Return one PDB path per unique SID, preferring the shortest filename (least dssp_edits)."""
    pdb_files = glob.glob(os.path.join(pdb_dir, '*.pdb'))
    sid_to_pdb = {}
    for pdb in pdb_files:
        sid = os.path.basename(pdb).split('_')[0]
        if sid not in sid_to_pdb or len(pdb) < len(sid_to_pdb[sid]):
            sid_to_pdb[sid] = pdb
    return sid_to_pdb


class CoordDataset(data.Dataset):
    """Dataset that loads pre-extracted coordinate .npy files and pads them."""

    def __init__(self, coords_dir, pad_len=15675):
        self.coord_files = sorted(glob.glob(os.path.join(coords_dir, '*_coords.npy')))
        self.pad_len = pad_len  # number of residue rows to pad to (must match trained model's input_shape)

    def __len__(self):
        return len(self.coord_files)

    def __getitem__(self, idx):
        path = self.coord_files[idx]
        basename = os.path.basename(path).replace('_coords.npy', '')
        coords = np.load(path).astype(np.float32)
        # Pad to (pad_len, 3) — DeepSetLayer expects (batch, n_residues, 3)
        if coords.shape[0] > self.pad_len:
            coords = coords[:self.pad_len]
        padded = np.zeros((self.pad_len, 3), dtype=np.float32)
        padded[:coords.shape[0]] = coords
        return basename, padded


@click.command()
@click.option('--pdb_dir', default='/home/sabari/ProteinSol/combining_geom_topo/data/train')
@click.option('--trained_model_dir', default='/home/sabari/ProteinSol/combining_geom_topo/results/topocoder_models')
@click.option('--pad_len', default=15675, help='Number of residue rows to pad to (must match trained model input_shape)')
@click.option('--batch_size', default=32)
@click.option('--num_workers', default=8)
def main(pdb_dir, trained_model_dir, pad_len, batch_size, num_workers):
    coords_dir = os.path.join(pdb_dir, 'coords')
    os.makedirs(coords_dir, exist_ok=True)

    # Step 1: Extract CA coordinates (one per unique SID)
    sid_to_pdb = get_unique_pdbs(pdb_dir)
    print(f"Found {len(sid_to_pdb)} unique SIDs in {pdb_dir}")

    existing_coords = set(
        f.replace('_coords.npy', '').split('_', 1)[1]
        for f in os.listdir(coords_dir) if f.endswith('_coords.npy')
    )
    to_extract = {sid: pdb for sid, pdb in sid_to_pdb.items() if sid not in existing_coords}
    print(f"Extracting coordinates for {len(to_extract)} PDBs ({len(existing_coords)} already exist)")

    for idx, (sid, pdb_path) in enumerate(tqdm(to_extract.items(), desc="Extracting CA coords")):
        try:
            coords = get_CA_coords(pdb_path)
            save_path = os.path.join(coords_dir, f"{idx}_{sid}_coords.npy")
            np.save(save_path, coords)
        except Exception as e:
            print(f"Error extracting coords for SID {sid}: {e}")

    # Step 2: Run Topocoder inference
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset = CoordDataset(coords_dir, pad_len=pad_len)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                 pin_memory=True, num_workers=num_workers)
    print(f"Running inference on {len(dataset)} coordinate files")

    for betti_no in [0, 1, 2]:
        print(f"\n--- Betti {betti_no} ---")
        topo_model = TopoCoder(
            input_shape=pad_len,
            output_shape=100,
            deepsets_shapes=[1200, 600, 300, 150, 50],
            dense_shapes=[50, 75, 100],
            use_bias=True,
            use_sigmoid=False,
        )
        ckpt_path = os.path.join(trained_model_dir, f'TopoCoderChkpt_Betti{betti_no}.pt')
        topo_model.load_state_dict(torch.load(ckpt_path, map_location=device)['state_dict'])
        topo_model.to(device)
        topo_model.eval()

        with torch.no_grad():
            for basenames, coords in tqdm(dataloader, desc=f"Betti {betti_no} inference"):
                coords = coords.to(device)
                emb = topo_model(coords)
                for i, name in enumerate(basenames):
                    sid = name.split('_', 1)[1]
                    save_path = os.path.join(pdb_dir, f"{name}_emb_b{betti_no}.pt")
                    torch.save(emb[i:i+1].cpu(), save_path)

    print(f"\nDone. Embeddings saved to {pdb_dir}")


if __name__ == "__main__":
    main()
