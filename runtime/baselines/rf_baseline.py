import glob
import json
import os
import time
import warnings

import click
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB.StructureBuilder import PDBConstructionWarning
from graphein.protein.resi_atoms import STANDARD_AMINO_ACIDS  # type: ignore
from graphein.utils.utils import onek_encoding_unk  # type: ignore
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

matplotlib.rcParams['figure.dpi'] = 300
plt.rcParams["font.family"] = 'Arial'
plt.rcParams.update({'font.size': 24})
plt.rcParams['svg.fonttype'] = 'none'

RANDOM_SEED = 42


def _get_dssp_features(pdb_path):
    """Extract AA one-hot (20) + RSA (1) + DSSP secondary structure (4) = 25 features per residue."""
    warnings.simplefilter('ignore', PDBConstructionWarning)
    p = PDBParser(QUIET=True)
    model = p.get_structure('prot', pdb_path)[0]
    dssp = DSSP(model, pdb_path)
    keys = list(dssp.keys())
    rsas = np.array([dssp[k][3] for k in keys]).reshape(-1, 1)
    aas = [dssp[k][1] for k in keys]
    aas_ohe = np.array([onek_encoding_unk(aa, STANDARD_AMINO_ACIDS) for aa in aas]).astype(float)
    # 20 AA one-hot + 1 RSA + 4 DSSP secondary structure features = 25
    return np.concatenate([aas_ohe, rsas], axis=-1)


def load_features(data_dir, csv_path, feature_source):
    """Load features and labels, returning (X, y, sids)."""
    sol_df = pd.read_csv(csv_path).set_index('sid')
    sids = sol_df.index.tolist()

    X_list = []
    y_list = []
    valid_sids = []
    errors = []

    for sid in tqdm(sids, desc=f'Loading {feature_source} features from {os.path.basename(data_dir)}'):
        try:
            if feature_source == 'esm':
                pt_path = os.path.join(data_dir, f'{sid}.pt')
                if not os.path.exists(pt_path):
                    continue
                emb = torch.load(pt_path, weights_only=True).cpu().numpy()
                # Mean pool across residues: (L, 1280) -> (1280,)
                features = emb.mean(axis=0)
            elif feature_source == 'graph':
                pt_path = os.path.join(data_dir, f'{sid}_input.pt')
                if not os.path.exists(pt_path):
                    continue
                graph = torch.load(pt_path, weights_only=False)
                # ndata['attr'] shape: (N, 1305, 1) -> squeeze -> (N, 1305) -> mean -> (1305,)
                attrs = graph.ndata['attr'].squeeze(-1).cpu().numpy()
                features = attrs.mean(axis=0)
            elif feature_source == 'full':
                esm_path = os.path.join(data_dir, f'{sid}.pt')
                pdb_matches = glob.glob(os.path.join(data_dir, f'{sid}_*_dssp_edits.pdb'))
                if not os.path.exists(esm_path) or not pdb_matches:
                    continue
                esm_emb = torch.load(esm_path, weights_only=True).cpu().numpy()
                dssp_feats = _get_dssp_features(pdb_matches[0])
                # Align lengths (DSSP and ESM may differ by a residue or two)
                min_len = min(len(esm_emb), len(dssp_feats))
                node_feats = np.concatenate([dssp_feats[:min_len], esm_emb[:min_len]], axis=-1)
                # Mean pool: (L, 1305) -> (1305,)
                features = node_feats.mean(axis=0)

            X_list.append(features)
            y_list.append(int(sol_df.loc[sid, 'solubility']))
            valid_sids.append(sid)
        except Exception as e:
            errors.append((sid, str(e)))

    X = np.array(X_list)
    y = np.array(y_list)
    if errors:
        print(f'{len(errors)} errors (first 3: {errors[:3]})')
    print(f'Loaded {len(X)} samples, feature dim={X.shape[1]}, '
          f'class balance: {y.sum()}/{len(y)} positive ({y.mean():.2%})')
    return X, y, valid_sids


@click.command()
@click.option('--train_dir', default='/home/sabari/ProteinSol/combining_geom_topo/data/train')
@click.option('--test_dir', default='/home/sabari/ProteinSol/combining_geom_topo/data/test')
@click.option('--train_csv', default='/home/sabari/ProteinSol/combining_geom_topo/data/csvs/training_set.csv')
@click.option('--test_csv', default='/home/sabari/ProteinSol/combining_geom_topo/data/csvs/test_set.csv')
@click.option('--feature_source', type=click.Choice(['graph', 'esm', 'full']), default='esm')
@click.option('--results_dir', default='/home/sabari/ProteinSol/combining_geom_topo/results/baselines')
@click.option('--sweep', is_flag=True, default=False, help='Run hyperparameter grid search')
@click.option('--n_estimators', default=500, type=int)
def run_rf_baseline(train_dir, test_dir, train_csv, test_csv,
                    feature_source, results_dir, sweep, n_estimators):
    print(f'Feature source: {feature_source}')
    X_train, y_train, _ = load_features(train_dir, train_csv, feature_source)
    X_test, y_test, test_sids = load_features(test_dir, test_csv, feature_source)

    if sweep:
        print('Running hyperparameter grid search (5-fold CV on train set)...')
        param_grid = {
            'n_estimators': [100, 300, 500],
            'max_depth': [None, 20, 50],
            'min_samples_leaf': [1, 3, 5],
        }
        grid = GridSearchCV(
            RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1),
            param_grid,
            cv=5,
            scoring='balanced_accuracy',
            verbose=1,
            n_jobs=1,  # RF already parallelized
        )
        grid.fit(X_train, y_train)
        clf = grid.best_estimator_
        print(f'Best params: {grid.best_params_}')
        print(f'Best CV balanced accuracy: {grid.best_score_:.4f}')
        best_params = grid.best_params_
    else:
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        print(f'Training RF with n_estimators={n_estimators}...')
        start = time.time()
        clf.fit(X_train, y_train)
        print(f'Training time: {time.time() - start:.1f}s')
        best_params = {'n_estimators': n_estimators}

    # Evaluate on test set
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print('\n' + '=' * 60)
    print(f'Random Forest Baseline ({feature_source} features)')
    print('=' * 60)
    print(f'Accuracy:          {acc:.4f}')
    print(f'Balanced Accuracy: {bal_acc:.4f}')
    print(f'F1 (macro):        {f1:.4f}')
    print(f'Precision (macro): {precision:.4f}')
    print(f'Recall (macro):    {recall:.4f}')
    print(f'ROC AUC:           {roc_auc:.4f}')
    print(f'\nConfusion Matrix:\n{cm}')
    print(f'\n{classification_report(y_test, y_pred, target_names=["insoluble", "soluble"])}')

    # Save confusion matrix plot
    os.makedirs(results_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["insoluble", "soluble"]).plot(ax=ax, cmap='Blues')
    ax.set_title(f'RF Baseline ({feature_source} features)')
    fig.tight_layout()
    cm_path = os.path.join(results_dir, f'rf_{feature_source}_confusion_matrix.svg')
    fig.savefig(cm_path)
    plt.close(fig)
    print(f'Confusion matrix saved to {cm_path}')

    # Save results
    results = {
        'model': 'RandomForest',
        'feature_source': feature_source,
        'feature_dim': X_train.shape[1],
        'n_train': len(X_train),
        'n_test': len(X_test),
        'params': best_params,
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'f1_macro': f1,
        'precision_macro': precision,
        'recall_macro': recall,
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist(),
    }
    results_path = os.path.join(results_dir, f'rf_{feature_source}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {results_path}')


if __name__ == '__main__':
    run_rf_baseline()
