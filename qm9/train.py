from typing import Union, List
from os import PathLike
from pathlib import Path
import subprocess

import pandas as pd

import utils

DATASET_PATH = utils.DATA_DIR.joinpath('processed/qm9-chemprop.csv')
LOG_DIR = Path('./logs')

CHEMPROP_PARAMS = dict(
        data_path=str(DATASET_PATH),
        dataset_type='regression',
        save_dir=str(LOG_DIR),
        smiles_column="smiles_relaxed",
        target_columns="gap",
        num_epoch=str(1),
)


def get_logdir() -> Path: 
    logdir = Path('./logs')
    if logdir.exists() is False: 
        logdir.mkdir()
    return logdir

def extract_chemprop_csv(
        df:pd.DataFrame,
        columns:List[str],
        savepath:Union[str, PathLike]=None, 
        force=False,
    ):
    """Format the QM9 dataset so that it can work with Chemprop. Expects SMILES to be in
    column 0.
    """
    # TODO: Create train/val/test splits here. Just return a single df for now.
    if Path(savepath).exists() is False or force is True: 
        _df = df.copy()
        _df.reindex(columns=columns)
        _df.to_csv(savepath)
    else:
        _df = pd.read_csv(savepath, index_col=0)
    return _df

def train_chemprop(
        data_path:str, 
        dataset_type:str, 
        save_dir:str,
        smiles_column:str, 
        target_columns:str,
        cache_cutoff=10000,
        num_workers=8,
        num_epoch=30,
    ):
    
    subprocess.call([
        "chemprop_train",
        "--data_path",
        f"{data_path}",
        "--dataset_type", 
        f"{dataset_type}",
        "--save_dir", 
        f"{save_dir}", 
        "--smiles_column", 
        f"{smiles_column}",
        "--target_columns", 
        f"{target_columns}",
        "--cache_cutoff", 
        f"{cache_cutoff}",
        "--num_workers",
        f"{num_workers}",
        "--epochs",
        f"{num_epoch}",
    ])



def main(): 
    """Train ChemProp Model"""

    if LOG_DIR.is_dir() is False: 
        LOG_DIR.mkdir()
    
    dataset = utils.get_qm9_dataset(force=False, rm_xyz=True)
    dataset = extract_chemprop_csv(
        dataset,
        columns=['smiles_relaxed', 'gap'], 
        savepath=DATASET_PATH,
        force=False,
    )
    train_chemprop(**CHEMPROP_PARAMS)

if __name__ == '__main__':
    main() 


