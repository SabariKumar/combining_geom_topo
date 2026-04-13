"""
Evaluate a saved Topoformer checkpoint on the test set and write a predictions CSV.

Output CSV columns:  ,preds_unrounded,targets,sids,preds,

Usage (from the combining_geom_topo root, inside the pixi environment):

    python runtime/topoformer/evaluate_test_set.py \
        --ckpt_path   results/47977508/topoformer_model.pth \
        --test_pdb_dir  data/test \
        --test_csv      data/csvs/test_set.csv \
        --barcode_dir   data/test \
        --output_csv    results/47977508/test_predictions.csv \
        [--num_degrees 4] [--num_channels 32] [--num_layers 7] \
        [--num_heads 8]    [--channels_div 2]  [--batch_size 8] \
        [--num_workers 4]  [--amp]
"""

import argparse
import os
import pathlib
import sys

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

# Allow imports relative to the combining_geom_topo root
_HERE = pathlib.Path(__file__).resolve().parent.parent.parent  # .../combining_geom_topo
sys.path.insert(0, str(_HERE))

from data_loading.topoformer.proteins import ProteinDataModule
from model.topoformer.fiber import Fiber
from model.topoformer.transformer import TopoformerPooled
from model.topoformer.utils import str2bool, using_tensor_cores


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Topoformer test-set evaluation")

    p.add_argument("--ckpt_path",    type=pathlib.Path, required=True,
                   help="Path to the saved .pth checkpoint")
    p.add_argument("--test_pdb_dir", type=pathlib.Path, required=True,
                   help="Directory containing test PDB files")
    p.add_argument("--test_csv",     type=pathlib.Path, required=True,
                   help="CSV with 'sid' and 'solubility' columns for test proteins")
    p.add_argument("--barcode_dir",  type=pathlib.Path, required=True,
                   help="Directory containing *_emb_b*.pt barcode embeddings")
    p.add_argument("--output_csv",   type=pathlib.Path, required=True,
                   help="Where to write the predictions CSV")

    # Model hyperparameters — must match those used during training
    p.add_argument("--num_degrees",  type=int, default=4)
    p.add_argument("--num_channels", type=int, default=32)
    p.add_argument("--num_layers",   type=int, default=7)
    p.add_argument("--num_heads",    type=int, default=8)
    p.add_argument("--channels_div", type=int, default=2)

    # Dataloader
    p.add_argument("--batch_size",   type=int, default=8)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--amp",          type=str2bool, nargs="?", const=True, default=False)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Build test dataloader
    # test_pdb_dir is used for both pdb_dir AND processed_dir (cache lives
    # next to the PDB files, same convention as training).
    # ------------------------------------------------------------------
    print("Building test dataset …")
    datamodule = ProteinDataModule(
        pdb_dir=str(args.test_pdb_dir),
        sol_df=str(args.test_csv),
        processed_dir=str(args.test_pdb_dir),
        use_barcodes=True,
        force_rebuild=False,
        barcode_dir=str(args.barcode_dir),
        # Pass the test set as the "external" split so that ds_test is the
        # whole test directory rather than a random slice of training data.
        external_test=str(args.test_pdb_dir),
        external_df=str(args.test_csv),
        external_barcode_dir=str(args.barcode_dir),
        external_esm_dir=str(args.test_pdb_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_degrees=args.num_degrees,
        amp=args.amp,
        precompute_bases=False,
    )
    test_loader = datamodule.test_dataloader()
    print(f"  Test set size: {len(datamodule.ds_test)} proteins")

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    print("Building model …")
    model = TopoformerPooled(
        output_dim=1,
        fiber_in=Fiber({0: datamodule.NODE_FEATURE_DIM}),
        fiber_out=Fiber({0: args.num_degrees * args.num_channels}),
        fiber_edge=Fiber({0: datamodule.EDGE_FEATURE_DIM}),
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
        amp=args.amp,
    )

    # ------------------------------------------------------------------
    # Load checkpoint
    # state_dict_module is the unwrapped (non-DDP) module state — use it
    # when running inference on a single GPU / CPU.
    # ------------------------------------------------------------------
    print(f"Loading checkpoint from {args.ckpt_path} …")
    ckpt = torch.load(str(args.ckpt_path), map_location=device)

    if "state_dict_module" in ckpt:
        model.load_state_dict(ckpt["state_dict_module"])
        print("  Loaded state_dict_module (unwrapped DDP weights)")
    elif "state_dict" in ckpt:
        # DDP checkpoint — strip the 'module.' prefix
        raw = ckpt["state_dict"]
        new_sd = {k.replace("module.", "", 1): v for k, v in raw.items()}
        model.load_state_dict(new_sd)
        print("  Loaded state_dict (stripped 'module.' prefix)")
    else:
        model.load_state_dict(ckpt)
        print("  Loaded raw state_dict")

    model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    print("Running inference …")
    rows = []

    with torch.inference_mode():
        for batch in tqdm(test_loader, unit="batch"):
            batch_list = list(batch)
            sids = batch_list.pop(0)          # first element is always sids
            batch = tuple(batch_list)
            *inputs, target = batch

            # Move tensors to device (graphs and dicts of tensors)
            cuda_inputs = []
            for x in inputs:
                if isinstance(x, torch.Tensor):
                    cuda_inputs.append(x.to(device))
                elif isinstance(x, dict):
                    cuda_inputs.append({k: v.to(device) for k, v in x.items()})
                else:
                    # DGL graph
                    cuda_inputs.append(x.to(device))
            target = target.to(device)

            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(*cuda_inputs)
                preds_unrounded = torch.sigmoid(logits).squeeze(-1)

            preds_unrounded_np = preds_unrounded.detach().cpu().numpy()
            targets_np = target.detach().cpu().numpy()

            for sid, pu, tgt in zip(sids, preds_unrounded_np, targets_np):
                rows.append({
                    "preds_unrounded": float(pu),
                    "targets": int(tgt),
                    "sids": sid,
                    "preds": int(round(float(pu))),
                })

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    df = pd.DataFrame(rows, columns=["preds_unrounded", "targets", "sids", "preds"])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(args.output_csv))   # default index gives the leading comma column
    print(f"Saved {len(df)} predictions to {args.output_csv}")

    # Quick summary
    correct = (df["preds"] == df["targets"]).sum()
    print(f"Accuracy: {correct}/{len(df)} = {correct/len(df):.4f}")


if __name__ == "__main__":
    main()
