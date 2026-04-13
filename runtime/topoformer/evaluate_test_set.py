"""
Evaluate a saved Topoformer checkpoint on the test set and write a predictions CSV.

Output CSV columns:  ,preds_unrounded,targets,sids,preds,

Usage (from the combining_geom_topo root, inside the pixi environment):

    python runtime/topoformer/evaluate_test_set.py \
        --ckpt_path     results/47977508/topoformer_model.pth \
        --test_pdb_dir  data/test \
        --test_csv      data/csvs/test_set.csv \
        --barcode_dir   data/test \
        --output_csv    results/47977508/test_predictions.csv \
        [--num_degrees 4] [--num_channels 32] [--num_layers 7] \
        [--num_heads 8]   [--channels_div 2]  [--batch_size 8] \
        [--num_workers 4] [--amp]
"""

import argparse
import pathlib
import sys

# ---------------------------------------------------------------------------
# Resolve repo root and add to path before any local imports
# ---------------------------------------------------------------------------
_HERE = pathlib.Path(__file__).resolve().parent.parent.parent  # .../combining_geom_topo
sys.path.insert(0, str(_HERE))

# ---------------------------------------------------------------------------
# CLI (parsed first so --help works without waiting for heavy imports)
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Topoformer test-set evaluation")
    p.add_argument("--ckpt_path",    type=pathlib.Path, required=True)
    p.add_argument("--test_pdb_dir", type=pathlib.Path, required=True,
                   help="Directory containing test PDB files AND cached *_input.pt / *_target.pt files")
    p.add_argument("--test_csv",     type=pathlib.Path, required=True,
                   help="CSV with 'sid' and 'solubility' columns")
    p.add_argument("--barcode_dir",  type=pathlib.Path, required=True,
                   help="Directory containing *_emb_b*.pt barcode embeddings")
    p.add_argument("--output_csv",   type=pathlib.Path, required=True)
    # Model hyperparameters — must match training
    p.add_argument("--num_degrees",  type=int, default=4)
    p.add_argument("--num_channels", type=int, default=32)
    p.add_argument("--num_layers",   type=int, default=7)
    p.add_argument("--num_heads",    type=int, default=8)
    p.add_argument("--channels_div", type=int, default=2)
    # Dataloader
    p.add_argument("--batch_size",   type=int, default=8)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--amp",          action="store_true", default=False)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Collate — replicates ProteinDataModule._collate for the no-precompute path
# ---------------------------------------------------------------------------

NODE_FEATURE_DIM = 25 + 1280   # amino-acid one-hot (25) + ESM-2 (1280)
EDGE_FEATURE_DIM = 8


def _get_relative_pos(graph):
    x = graph.ndata["pos"]
    src, dst = graph.edges()
    return x[dst] - x[src]


def collate(samples):
    import dgl
    import torch

    graphs, barcodes, targets, sids = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    batched_graph.edata["rel_pos"] = _get_relative_pos(batched_graph)

    node_feats  = {"0": batched_graph.ndata["attr"][:, :NODE_FEATURE_DIM].float()}
    edge_feats  = {"0": batched_graph.edata["edge_attr"][:, :EDGE_FEATURE_DIM, None].float()}
    barcode_feats = {"0": torch.stack(barcodes, dim=0).float()}
    target_tensor = torch.tensor(targets)

    return sids, batched_graph, node_feats, barcode_feats, edge_feats, target_tensor


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("Importing torch …", flush=True)
    import torch
    print("Importing dgl …", flush=True)
    import dgl  # noqa: F401 — needed so dgl is available inside collate
    print("Importing pandas …", flush=True)
    import pandas as pd
    print("Importing tqdm …", flush=True)
    from tqdm import tqdm
    from torch.utils.data import DataLoader

    print("Importing local modules …", flush=True)
    from data_loading.topoformer.protein_dataset import ProteinDataset
    from model.topoformer.fiber import Fiber
    from model.topoformer.transformer import TopoformerPooled
    from model.topoformer.utils import using_tensor_cores

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # ------------------------------------------------------------------
    # Dataset & dataloader  (ProteinDataset directly — no ProteinDataModule)
    # ------------------------------------------------------------------
    print("Building test dataset …", flush=True)
    dataset = ProteinDataset(
        pdb_dir=str(args.test_pdb_dir),
        sol_df=str(args.test_csv),
        mode="test",
        use_barcodes=True,
        processed_dir=str(args.test_pdb_dir),
        barcode_dir=str(args.barcode_dir),
        esm_dir=str(args.test_pdb_dir),
        force_rebuild=False,
    )
    print(f"  Test set size: {len(dataset)} proteins", flush=True)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print("Building model …", flush=True)
    model = TopoformerPooled(
        output_dim=1,
        fiber_in=Fiber({0: NODE_FEATURE_DIM}),
        fiber_out=Fiber({0: args.num_degrees * args.num_channels}),
        fiber_edge=Fiber({0: EDGE_FEATURE_DIM}),
        tensor_cores=using_tensor_cores(args.amp),
        comb_type="attn",
        use_topo_projection=False,
        save_feats_dir=str(args.output_csv.parent),
        run_id="eval",
        num_degrees=args.num_degrees,
        num_channels=args.num_channels,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        channels_div=args.channels_div,
        pooling=None,   # required by TopoformerPooled.__init__; None → 'max'
        amp=args.amp,
    )

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    print(f"Loading checkpoint from {args.ckpt_path} …", flush=True)
    ckpt = torch.load(str(args.ckpt_path), map_location=device)

    if "state_dict_module" in ckpt:
        model.load_state_dict(ckpt["state_dict_module"])
        print("  Loaded state_dict_module (unwrapped DDP weights)", flush=True)
    elif "state_dict" in ckpt:
        raw = ckpt["state_dict"]
        new_sd = {k.replace("module.", "", 1): v for k, v in raw.items()}
        model.load_state_dict(new_sd)
        print("  Loaded state_dict (stripped 'module.' prefix)", flush=True)
    else:
        model.load_state_dict(ckpt)
        print("  Loaded raw state_dict", flush=True)

    model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    print("Running inference …", flush=True)
    rows = []

    with torch.inference_mode():
        for batch in tqdm(loader, unit="batch"):
            sids, graph, node_feats, barcode_feats, edge_feats, target = batch

            graph        = graph.to(device)
            node_feats   = {k: v.to(device) for k, v in node_feats.items()}
            barcode_feats= {k: v.to(device) for k, v in barcode_feats.items()}
            edge_feats   = {k: v.to(device) for k, v in edge_feats.items()}
            target       = target.to(device)

            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(graph, node_feats, barcode_feats, edge_feats)
                preds_unrounded = torch.sigmoid(logits).squeeze(-1)

            for sid, pu, tgt in zip(
                sids,
                preds_unrounded.detach().cpu().tolist(),
                target.detach().cpu().tolist(),
            ):
                rows.append({
                    "preds_unrounded": pu,
                    "targets": int(tgt),
                    "sids": sid,
                    "preds": int(round(pu)),
                })

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    df = pd.DataFrame(rows, columns=["preds_unrounded", "targets", "sids", "preds"])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(args.output_csv))
    print(f"Saved {len(df)} predictions to {args.output_csv}", flush=True)

    correct = (df["preds"] == df["targets"]).sum()
    print(f"Accuracy: {correct}/{len(df)} = {correct/len(df):.4f}", flush=True)


if __name__ == "__main__":
    main()
