import glob
import math
import os
from typing import List, Optional, Union

# Plotting length dist.
import matplotlib
import numpy as np
import torch
import torch.utils.data as data
from matplotlib import pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, random_split

matplotlib.rcParams["figure.dpi"] = 300
plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({"font.size": 24})
plt.rcParams["svg.fonttype"] = "none"

print(f"Using torch version: {torch.__version__}")
RANDOM_SEED = 42
torch.backends.cudnn.deterministic = True
generator = torch.Generator().manual_seed(RANDOM_SEED)


class TopoCoderDataset(Dataset):
    def __init__(
        self, input_type, labels_path, data_dir, return_labels: bool = False
    ) -> None:
        self.input_type = input_type
        self.labels_path = labels_path
        self.data_dir = data_dir
        self.return_labels = return_labels
        self.labels = np.load(labels_path).astype(np.float32)
        self.max_seq_len, _ = self._get_max_len()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.toList()
        if self.input_type == "pi":
            pi = np.load(
                glob.glob(os.path.join(self.data_dir, str(idx) + "_*_pi.npy"))[0]
            ).astype(np.float32)
        else:
            pi = np.load(
                glob.glob(os.path.join(self.data_dir, str(idx) + "_*_vec.npy"))[0]
            ).astype(np.float32)
        coords = np.load(
            glob.glob(os.path.join(self.data_dir, str(idx) + "_*_coords.npy"))[0]
        ).astype(np.float32)
        coords = self._pad_coords(coords)
        if self.return_labels:
            sample = {"pi": pi, "coords": coords, "label": self.labels[idx]}
        else:
            sample = {"pi": pi, "coords": coords}
        return sample

    def _get_max_len(self):
        print(
            f"Getting max sequence length for coordinate zero padding at {self.data_dir}..."
        )
        coord_files = glob.glob(os.path.join(self.data_dir, "*_coords.npy"))
        lens = []
        for coord in coord_files:
            coord_ = np.load(coord)
            lens.append(coord_.shape[0])
        print(f"Max sequence length: {max(lens)}")
        return max(lens), lens

    def _pad_coords(self, input_coords):
        targ_shape = (self.max_seq_len, 3)
        pad_width = [(0, j - i) for i, j in zip(input_coords.shape, targ_shape)]
        return np.pad(input_coords, pad_width)

    def plot_seq_len_distribution(self):
        _, lens = self._get_max_len()
        bins = np.linspace(math.ceil(min(lens)), math.floor(max(lens)), 40)
        plt.xlim([min(lens) - 4, max(lens) + 4])

        plt.hist(lens, bins=bins, alpha=0.5)
        plt.title("Protein Lengths")
        plt.xlabel("Amino Acid Count")
        plt.ylabel("No. of Samples")

        plt.savefig(os.path.join(self.data_dir, "amino_acid_length_distribution.svg"))


def pad_coords(data):
    coords = [torch.tensor(d["coords"]) for d in data]
    pis = [d["pi"] for d in data]  # all uniform size
    coords = pad_sequence(coords, batch_first=True)
    pis = torch.tensor(pis)
    return {"pi": pis, "coords": coords}


def get_split_lens(proportions, dataset):
    lengths = [int(p * len(dataset)) for p in proportions]
    lengths[-1] = len(dataset) - sum(lengths[:-1])
    return lengths


def make_dataloaders(
    input_type: str,
    betti_no: List[int],
    train_pi_dir: Union[str, bytes, os.PathLike],
    labels_path: Union[
        str, bytes, os.PathLike
    ] = "/home/sabari/ProteinSol/combining_geom_topo/data/topocoder_labels/1_normalization/labels.npy",
    splits: Optional[List[float]] = [0.8, 0.1, 0.1],
    batch_size: int = 1000,
    shuffle_train: bool = True,
    pin_memory: bool = True,
    num_workers: int = 32,
):
    # Function to generate PI Dataloaders with deterministic splits, or just one big dataset (for debug)
    loader_dict = {}
    for betti_ in betti_no:
        data_dir = str(os.path.join(train_pi_dir, f"betti_{betti_}"))
        data_set = TopoCoderDataset(
            input_type=input_type, labels_path=labels_path, data_dir=data_dir
        )
        data_set.plot_seq_len_distribution()
        if splits:
            split_lens = get_split_lens(splits, data_set)
            train_set, val_set, test_set = random_split(
                data_set, split_lens, generator=generator
            )
            train_loader = data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=shuffle_train,
                pin_memory=pin_memory,
                num_workers=num_workers,
            )
            val_loader = data.DataLoader(
                val_set,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=pin_memory,
                num_workers=num_workers,
            )
            test_loader = data.DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=pin_memory,
                num_workers=num_workers,
            )
            loader_dict[betti_] = (train_loader, val_loader, test_loader)
        else:
            loader_dict[betti_] = data.DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=pin_memory,
                num_workers=num_workers,
            )
    return loader_dict


class TopoCoderInferenceDataset(Dataset):
    # Utility class for running batched inference to generate embeddings
    def __init__(self, 
                 coords_dir: Union[str, bytes, os.PathLike] = '/home/sabari/ProteinSol/combining_geom_topo/data/topocoder_labels/1_normalization/betti_0',
                 pad_len: int = 15675):
        self.coords = glob.glob(os.path.join(coords_dir,  "*_coords.npy"))
        self.max_seq_len = pad_len

    def __len__(self):
        return len(self.coords)
    
    def __getitem__(self, idx):
        basename = '_'.join(os.path.basename(self.coords[idx]).split('_')[:2])
        coords = np.load(self.coords[idx])
        coords = self._pad_coords(coords)
        return (basename, coords)
    
    def _pad_coords(self, input_coords):
        targ_shape = (self.max_seq_len, 3)
        pad_width = [(0, j - i) for i, j in zip(input_coords.shape, targ_shape)]
        return np.pad(input_coords, pad_width)