# Enthought product code
#
# (C) Copyright 2010-2023 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This file and its contents are confidential information and NOT open source.
# Distribution is prohibited.

from typing import Union, List
from os import PathLike
from pathlib import Path
import subprocess
from urllib.request import urlretrieve
import tarfile
from tqdm import tqdm

import pandas as pd 

# config
LOCAL = Path(__file__).parent.resolve() 
ROOT_DIR = LOCAL.parent.resolve() 
DATA_DIR = LOCAL.joinpath("datasets").resolve() 

# data managment

def get_qm9_dataset(force:bool=False, rm_xyz=True) -> pd.DataFrame: 
    """Download and prepare the QM9 dataset."""
    
    URL = "https://figshare.com/ndownloader/files/3195389"
    PROCESSED_DATA_NAME = DATA_DIR / 'processed/qm9.csv'
    property_labels = [
        "UID", #     -            Consecutive, 1-based integer identifier of molecule
        "A", #         GHz          Rotational constant A
        "B", #         GHz          Rotational constant B
        "C", #         GHz          Rotational constant C
        "mu", #        Debye        Dipole moment
        "alpha", #     Bohr^3       Isotropic polarizability
        "homo", #      Hartree      Energy of Highest occupied molecular orbital (HOMO)
        "lumo", #      Hartree      Energy of Lowest occupied molecular orbital (LUMO)
        "gap", #       Hartree      Gap, difference between LUMO and HOMO
        "r2", #        Bohr^2       Electronic spatial extent
        "zpve", #      Hartree      Zero point vibrational energy
        "U0", #        Hartree      Internal energy at 0 K
        "U", #         Hartree      Internal energy at 298.15 K
        "H", #         Hartree      Enthalpy at 298.15 K
        "G", #         Hartree      Free energy at 298.15 K
        "Cv", #        cal/(mol K)  Heat capacity at 298.15 K
    ] 

    # download and pre-processing
    if PROCESSED_DATA_NAME.exists() is False or force is True: 
        prepare_data_dir() 
        urlretrieve(URL, DATA_DIR / 'raw/qm9.tar.gz')
        extract_tarfile(DATA_DIR / 'raw/qm9.tar.gz', DATA_DIR/'raw')
        
        data = [] 
        for xyz in tqdm(Path(DATA_DIR / 'raw').glob('*.xyz')): 
            with open(xyz) as fh: 
                lines = fh.readlines()

            n_atoms = int(lines[0].strip()) 
            smiles = lines[-2].strip().split()
            inchi = lines[-1].strip().split() 
            properties = lines[1].split()[1:]
            data.append(properties + [n_atoms] + smiles + inchi ) 
            if rm_xyz: 
                xyz.unlink()
        
        # collect data into dataframe
        data = pd.DataFrame(
            data,
            columns=property_labels + ['n_atoms', 'smiles_raw','smiles_relaxed', "inchi_raw", "inchi_relaxed"]
        )
        data.iloc[:,0] = data.iloc[:,0].astype(int)
        data.iloc[:,1:16] = data.iloc[:,1:16].astype(float)
        data.to_csv(PROCESSED_DATA_NAME)
        return data
    else: 
        data = pd.read_csv(PROCESSED_DATA_NAME, index_col=0)
        return data

def extract_tarfile(
        tar_filename:Union[str, PathLike], extract_path:Union[str, PathLike]
    ) -> None:
    """Unpack a tar file."""
    try:
        with tarfile.open(tar_filename, 'r') as tar:
            tar.extractall(path=extract_path)
        print(f"Successfully extracted contents from {tar_filename} to {extract_path}")
    except tarfile.TarError as e:
        print(f"Error extracting {tar_filename}: {e}")

def prepare_data_dir() -> None: 
    """Create the required subdirs in the dataset directory."""
    for dirname in ['raw', 'processed']: 
        subdir = DATA_DIR.joinpath(dirname)
        if subdir.exists() is False: 
            subdir.mkdir(parents=True)

if __name__ == '__main__':
    print('Config:')
    print(f"\t{LOCAL}")
    print(f"\t{ROOT_DIR}")
    print(f"\t{DATA_DIR}")
