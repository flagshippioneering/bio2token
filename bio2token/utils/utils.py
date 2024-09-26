import yaml
import os
from Bio.PDB import PDBParser
import numpy as np
import torch
from typing import List


AMINO_ACID_ABBRS = {
    "A": "ALA",
    "B": "ASX",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "U": "SEC",
    "V": "VAL",
    "W": "TRP",
    "X": "XAA",
    "Y": "TYR",
    "Z": "GLX",
}
# reversed amino acid abbriviations
AMINO_ACID_ABBRS_REVERSED = {v: k for k, v in AMINO_ACID_ABBRS.items()}


class Config:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            if isinstance(v, dict):
                if k == "ssm_cfg":
                    setattr(self, k, None)
                else:
                    setattr(self, k, Config(v))
            else:
                # print(k, v)
                setattr(self, k, v)


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return Config(config)


def pdb_2_dict(pdb_path: str, chains: List[str] = None):
    parser = PDBParser()
    pdb_ids = []
    pdb_id = os.path.basename(pdb_path).split(".")[0]
    structure = parser.get_structure(pdb_id, pdb_path)
    seq = ""
    coords = []
    atom_names = []
    atom_types = []
    res_names = []
    chain_ids = []
    # if no chains are provided we read in everything
    if chains is None:
        chains = [chain.id for chain in structure.get_chains()]

    for chain in structure.get_chains():
        if chain.id not in chains:
            pass
        else:
            for residue in chain:
                resname = residue.get_resname()
                if resname not in AMINO_ACID_ABBRS_REVERSED:
                    pass
                else:
                    res = AMINO_ACID_ABBRS_REVERSED[resname]
                    seq += res
                    for atom in residue:
                        atom_names.append(atom.get_name())
                        atom_types.append(atom.get_name()[0])
                        chain_ids.append(chain.id)
                        pdb_ids.append(pdb_id)
                        coords.append(np.array(atom.get_coord()))
                        res_names.append(res)

    coords = np.vstack(coords)
    # Create DataFrame
    pdb_dict = {
        "pdb_id": pdb_ids,
        "seq": seq,
        "res_names": res_names,
        "coords_groundtruth": coords,
        "atom_names": atom_names,
        "atom_types": atom_types,
        "seq_length": len(seq),
        "atom_length": len(atom_names),
        "chains": chain_ids,
    }
    return pdb_dict


def pdb_to_batch(config: Config, pdb_dict: dict):
    coords = pdb_dict["coords_groundtruth"]
    if config.recenter:
        barycenter = coords.mean(axis=0, keepdims=True)[0]
        if np.isnan(barycenter).any():
            raise (f"Found NaN in barycenter: {barycenter}")
        coords = coords - barycenter
    batch = {}
    coords_groundtruth = torch.zeros((1, config.max_len, 3))
    coords_groundtruth[:, : pdb_dict["atom_length"], :] = torch.FloatTensor(coords).unsqueeze(0)
    batch["coords_groundtruth"] = coords_groundtruth.cuda()
    input_mask = torch.zeros(1, config.max_len).bool()
    input_mask[:, : pdb_dict["atom_length"]] = True
    batch["atom_mask"] = input_mask.cuda()
    batch["seq_length"] = pdb_dict["atom_length"]
    return batch
