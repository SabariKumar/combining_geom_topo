import click
import os
import glob
from typing import Union
import numpy as np
import numpy.typing as npt
from Bio.PDB import *
from tqdm import tqdm

def get_CA_alphafold(
    input_path: Union[str, bytes, os.PathLike],
) -> npt.NDArray[np.float32]:
    # Returns an (n_amino_acids x 3 numpy array of alpha carbon coordinates)
    p = PDBParser(QUIET=True)
    structure = p.get_structure("input", input_path)
    for model in structure:
        for chain in model:
            CA_coords = []
            for residue in chain:
                CA_coords.append(residue["CA"].get_coord())
    return np.array(CA_coords, dtype="float32")

@click.command()
@click.option('--pdb_dir', required = True)
@click.option('--save_dir', required = True)
def make_coords(pdb_dir, save_dir):
    pdb_list = glob.glob(os.path.join(pdb_dir, '*.pdb'))
    for ind, pdb in tqdm(enumerate(pdb_list)):
        coords = get_CA_alphafold(pdb)
        sid = os.path.basename(pdb).split('_')[0]
        np.save(os.path.join(save_dir, str(ind) + '_' + str(sid)+ '_coords.npy'), coords)

if __name__ == '__main__':
    make_coords()