# Training script for topocoder models. Does not use lightning for
# full compatibility with the original Topoformer docker +singularity
# containers

import os
from typing import List, Optional, Union

import click  # type: ignore
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import wandb
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split

import sys

sys.path.append("/home/sabari/ProteinSol/topoformer")

from data_loading.topocoder.topocoder_loader import make_dataloaders
from model.topocoder.topocoder import TopoCoder

print(f"Using torch version: {torch.__version__}")
RANDOM_SEED = 42
torch.backends.cudnn.deterministic = True
generator = torch.Generator().manual_seed(RANDOM_SEED)


def train_epoch(
    train_loader: data.DataLoader,
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss: str,
    per_batch_log_interval: Optional[int] = None,
):
    l1_running_loss = 0
    l2_running_loss = 0
    for batch_idx, sample in enumerate(train_loader):
        optimizer.zero_grad()
        preds = model(sample["coords"])
        l1_loss = F.l1_loss(preds, sample["pi"])
        l2_loss = F.mse_loss(preds, sample["pi"])
        if loss == 'l1':
            l1_loss.backward()
        elif loss == 'l2':
            l2_loss.backward()
        else:
            raise NotImplementedError('Loss must be either l1 or l2') 
        optimizer.step()
        l1_running_loss += l1_loss.item()
        l2_running_loss += l2_loss.item()
        if per_batch_log_interval:
            if batch_idx % per_batch_log_interval == 0:
                wandb.log({"per_batch_train_l1_loss": l1_running_loss})
                wandb.log({"per_batch_train_l2_loss": l2_running_loss})
    return (l1_running_loss / len(train_loader), l2_running_loss / len(train_loader))


def train_topocoder(
    betti_no: int,
    # input_type: str, Deprecated!
    train_loader,
    valid_loader,
    checkpoint_dir: Union[str, bytes, os.PathLike],
    input_dim: int = 15675,
    output_dim: int = 100,
    device: str = "cuda:1",
    lr: float = 5e-3,
    n_epochs: int = 100,
    log_interval: int = 1,
    per_batch_log_interval: Optional[int] = None,
    valid_interval: int = 5,
    loss: str = 'l1',
    **kwargs,
):
    device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using torch device: {device}")
    root_dir = os.path.join(checkpoint_dir, f"BettiNo{betti_no}")
    os.makedirs(root_dir, exist_ok=True)
    model = TopoCoder(
        input_shape=input_dim,
        output_shape=output_dim,
        deepsets_shapes=[1200, 600, 300, 150, 50],
        dense_shapes=[50, 75, 100],
        use_bias=True,
        use_sigmoid=False,
    )
    optimizer = optim.Adamax(model.parameters(), lr=lr)
    for epoch in range(n_epochs):
        model.train(True)
        l1_loss, l2_loss = train_epoch(
            train_loader=train_loader,
            model=model,
            optimizer=optimizer,
            per_batch_log_interval=per_batch_log_interval,
            loss = loss
        )
        if epoch % log_interval == 0:
            wandb.log({f"{betti_no}_train_l1_loss": l1_loss})
            wandb.log({f"{betti_no}_train_l2_loss": l2_loss})
        print(f'Epoch {epoch}: l1 loss {l1_loss}, l2 loss {l2_loss}\n')

        if epoch % valid_interval:
            valid_l1_running_loss = 0
            valid_l2_running_loss = 0
            model.eval()
            with torch.no_grad():
                for i, valid_sample in enumerate(valid_loader):
                    valid_preds = model(valid_sample["coords"])
                    valid_l1_loss = F.l1_loss(valid_preds, valid_sample["pi"])
                    valid_l2_loss = F.mse_loss(valid_preds, valid_sample["pi"])
                    valid_l1_running_loss += valid_l1_loss.item()
                    valid_l2_running_loss += valid_l2_loss.item()
            wandb.log(
                {f"{betti_no}_valid_l1_loss": valid_l1_running_loss / len(valid_loader)}
            )
            wandb.log(
                {f"{betti_no}_valid_l2_loss": valid_l2_running_loss / len(valid_loader)}
            )
            print(f'Epoch {epoch}: validation l1 loss {valid_l1_loss/len(valid_loader)}, validation l2 loss {valid_l2_loss/len(valid_loader)}\n')


    return model, {f'{betti_no}_train_l1_loss': l1_loss,
                   f'{betti_no}_valid_l1_loss': valid_l1_loss}

def eval_topocoder(betti_no, model, test_loader):
    # Generate test set stats
    test_l1_running_loss = 0
    test_l2_running_loss = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            test_preds = model(sample["coords"])
            test_l1_loss = F.l1_loss(test_preds, sample["pi"])
            test_l2_loss = F.mse_loss(test_preds, sample["pi"])
            test_l1_running_loss += test_l1_loss.item()
            test_l2_running_loss += test_l2_loss.item()
        wandb.log({f"{betti_no}_test_l1_loss": test_l1_running_loss / len(test_loader)})
        wandb.log({f"{betti_no}_test_l2_loss": test_l2_running_loss / len(test_loader)})


@click.command()
@click.option("--betti_no", required=True, multiple=True)
@click.option("--input_type", required=True)
@click.option("--train_pi_dir", required=True)
@click.option("--checkpoint_dir", required=True)
@click.option("--model_save_dir", required=True)
@click.option("--device", default="cuda:1")
@click.option("--test_set_inf", default=True)
@click.option("--lr", default=5e-3)
@click.option("--n_epochs", default=100)
@click.option("--batch_size", default = 1000)
@click.option("--loss", default = 'l1')
def topo_runner(
    betti_no,
    input_type,
    train_pi_dir,
    checkpoint_dir,
    model_save_dir,
    device,
    test_set_inf,
    lr,
    n_epochs,
    batch_size,
    loss
):
    betti_no = [int(x) for x in betti_no]
    loader_dict = make_dataloaders(
        input_type=input_type, betti_no=betti_no, train_pi_dir=train_pi_dir, batch_size=batch_size
    )
    for betti_ in betti_no:
        train_loader, valid_loader, test_loader = loader_dict[betti_]
        wandb.init(
            project="TopoCoderEmbeddings",
            tags=[f"BettiNo{betti_}"],
            config={
                "betti_no": betti_,
                "input_type": input_type,
                "train_pi_dir": train_pi_dir,
                "checkpoint_dir": checkpoint_dir,
                "model_save_dir": model_save_dir,
                "lr": lr,
            },
        )
        topo_model, topo_result = train_topocoder(
            betti_no=betti_,
            # input_type = input_type, Deprecated!
            train_loader=train_loader,
            valid_loader=valid_loader,
            checkpoint_dir=checkpoint_dir,
            device=device,
            input_dim=15675,
            output_dim=100,
            use_bias=True,
            dropout=0.0,
            input_dropout=0.0,
            lr=lr,
            n_epochs=n_epochs,
            loss = loss
        )
        eval_topocoder(betti_, topo_model, test_loader)
        try:
            state_dict = topo_model.state_dict()
            checkpoint = {"state_dict": state_dict, "model_results": topo_result}
            torch.save(
                checkpoint,
                os.path.join(model_save_dir, f"TopoCoderChkpt_Betti{betti_}.pt"),
            )

            if test_set_inf:
                # inf_model = topo_model.load_from_checkpoint(os.path.join(model_save_dir, f'TopoCoderChkpt_Betti{betti_no}.pt'))
                topo_model.eval()

                with torch.no_grad():
                    for sample in test_loader:
                        preds = topo_model(sample["coords"])
                        torch.save(
                            preds,
                            os.path.join(
                                model_save_dir, f"TopoCoderPreds_Betti{betti_}.pt"
                            ),
                        )

        except Exception as error:
            print("An error occurred:", type(error).__name__, "–", error)
            print("Unable to save checkpoint dict!")


if __name__ == "__main__":
    topo_runner()
