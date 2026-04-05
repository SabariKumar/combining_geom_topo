import os

import click  # type: ignore
import torch
import torch.utils.data as data

import sys
sys.path.append("/home/sabari/ProteinSol/combining_geom_topo")

from data_loading.topocoder.topocoder_loader import TopoCoderInferenceDataset
from model.topocoder.topocoder import TopoCoder

@click.command()
@click.option('--pdb_dir', default = '/home/sabari/ProteinSol/topoformer/data/soluprotgeom/processed/train/coords')
@click.option('--emb_save_dir', default = '/home/sabari/ProteinSol/topoformer/data/soluprotgeom/processed/train/embeddings')
@click.option("--betti_no", required=True, multiple=True)
@click.option("--trained_model_dir", default = '/home/sabari/ProteinSol/topoformer/results/')
def make_topo_emb(pdb_dir, emb_save_dir, betti_no, trained_model_dir):
    data_set = TopoCoderInferenceDataset(coords_dir = pdb_dir)
    data_loader = data.DataLoader(data_set,batch_size=1,
                shuffle=False,
                pin_memory=True,
                num_workers=8)
    betti_nos = [int(x) for x in betti_no]
    for betti_no in betti_nos:
        topo_model = TopoCoder(
            input_shape=15675,
            output_shape=100,
            deepsets_shapes=[1200, 600, 300, 150, 50],
            dense_shapes=[50, 75, 100],
            use_bias=True,
            use_sigmoid=False)
        topo_model.load_state_dict(torch.load(
            os.path.join(trained_model_dir, f'TopoCoderChkpt_Betti{betti_no}.pt'))['state_dict'])
        # Remove trailing ReLU after final Linear — it zeroes all-negative outputs
        topo_model.topo_net = torch.nn.Sequential(*list(topo_model.topo_net.children())[:-1])
        topo_model.eval()
        with torch.no_grad():
            for batch_idx, sample in enumerate(data_loader):
                filename = os.path.join(emb_save_dir, '_'.join([sample[0][0], 'emb', f'b{betti_no}.pt']))
                emb = topo_model(sample[1])
                torch.save(emb, filename)

if __name__ == "__main__":
    make_topo_emb()
