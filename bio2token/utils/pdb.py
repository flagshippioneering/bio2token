import numpy as np
from Bio.PDB import Structure, Model, Chain, Residue, Atom
from typing import List, Optional
from Bio.PDB import PDBIO, PDBParser
from Bio import PDB
from .utils import Config
import os
import torch

BB_ATOMS_AA = ["N", "CA", "C", "O"]
SC_ATOMS_AA = {
    "ALA": ["CB"],
    "ARG": ["CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["CB", "CG", "OD1", "ND2"],
    "ASP": ["CB", "CG", "OD1", "OD2"],
    "CYS": ["CB", "SG"],
    "GLN": ["CB", "CG", "CD", "OE1", "NE2"],
    "GLU": ["CB", "CG", "CD", "OE1", "OE2"],
    "GLY": [],
    "HIS": ["CB", "CG", "ND1", "CE1", "NE2", "CD2"],
}

BB_ATOMS_RNA = ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"]
SC_ATOMS_AA = {
    "ALA": ["CB"],
    "ARG": ["CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["CB", "CG", "OD1", "ND2"],
    "ASP": ["CB", "CG", "OD1", "OD2"],
    "CYS": ["CB", "SG"],
    "GLN": ["CB", "CG", "CD", "OE1", "NE2"],
    "GLU": ["CB", "CG", "CD", "OE1", "OE2"],
    "GLY": [],
    "HIS": ["CB", "CG", "ND1", "CE1", "NE2", "CD2"],
    "ILE": ["CB", "CG1", "CG2", "CD1"],
    "LEU": ["CB", "CG", "CD1", "CD2"],
    "LYS": ["CB", "CG", "CD", "CE", "NZ"],
    "MET": ["CB", "CG", "SD", "CE"],
    "PHE": ["CB", "CG", "CD1", "CE1", "CZ", "CE2", "CD2"],
    "PRO": ["CB", "CG", "CD"],
    "SER": ["CB", "OG"],
    "THR": ["CB", "OG1", "CG2"],
    "TRP": ["CB", "CG", "CD1", "NE1", "CE2", "CZ2", "CH2", "CZ3", "CE3", "CD2"],
    "TYR": ["CB", "CG", "CD1", "CE1", "CZ", "OH", "CE2", "CD2"],
    "VAL": ["CB", "CG1", "CG2"],
    "XAA": [],
}

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

RNA_ABBRS = {
    "A": "A",
    "G": "G",
    "C": "C",
    "U": "U",
    "T": "T",
}

BB_ATOMS_RNA = ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"]
SC_ATOMS_RNA = {
    "A": ["N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],
    "G": ["N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],
    "C": ["N1", "C2", "O2", "C6", "C5", "N4", "N3", "C4"],
    "U": ["N1", "C2", "O2", "C6", "C5", "N3", "C4", "O4"],
    "T": ["N1", "C2", "O2", "C6", "C5", "C7", "O4", "N3", "C4"],  # Thymine for DNA
}
# reversed dict
AA_ABRV_REVERSED = {v: k for k, v in AMINO_ACID_ABBRS.items()}
RNA_ABRV_REVERSED = {v: k for k, v in RNA_ABBRS.items()}


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
    res_ids = []
    res_types = []
    continuous_res_ids = []
    # if no chains are provided we read in everything
    if chains is None:
        chains = [chain.id for chain in structure.get_chains()]
    continuous_res_id = 1
    for chain in structure.get_chains():
        if chain.id not in chains:
            pass
        else:
            res_id = 1
            for residue in chain:
                resname = residue.get_resname()
                if len(resname) == 3:
                    ABBRS_REVERSED = AA_ABRV_REVERSED
                    res_type = "aa"
                else:
                    ABBRS_REVERSED = RNA_ABRV_REVERSED
                    res_type = "rna"
                if resname not in ABBRS_REVERSED:
                    pass
                else:
                    res = ABBRS_REVERSED[resname]
                    seq += res
                    for atom in residue:
                        atom_name = atom.get_name()
                        if atom_name[0] == "H":
                            pass
                        else:
                            atom_names.append(atom.get_name())
                            atom_types.append(atom.get_name()[0])
                            chain_ids.append(chain.id)
                            pdb_ids.append(pdb_id)
                            coords.append(np.array(atom.get_coord()))
                            res_names.append(res)
                            res_ids.append(res_id)
                            continuous_res_ids.append(continuous_res_id)
                            res_types.append(res_type)
                            res_id += 1
                    continuous_res_id += 1
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
        "res_ids": res_ids,
        "continuous_res_ids": continuous_res_ids,
        "res_types": res_types,
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


def structure_to_pdb(struct: Structure.Structure, pdb_file_path: str):
    # Write the structure to a PDB file
    io = PDBIO()
    io.set_structure(struct)
    io.save(pdb_file_path)


def pdb_dict_to_file(pdb_dict: dict, bb_only: bool = False, pdb_file_path: str = None) -> Structure.Structure:

    # Create a new structure
    structure = Structure.Structure("")
    # Create a new model
    model = Model.Model(0)
    # Create a new chain
    chain = Chain.Chain("A")
    # Create a new atom for each backbone coordinate
    counter = 1
    for id in pdb_dict.keys():
        res_type = pdb_dict[id]["res_types"][0]
        if res_type == "aa":
            BB_ATOMS = BB_ATOMS_AA
            SC_ATOMS = SC_ATOMS_AA
            ABBR = AMINO_ACID_ABBRS
            BB_LEN = 4
        elif res_type == "rna":
            BB_ATOMS = BB_ATOMS_RNA
            SC_ATOMS = SC_ATOMS_RNA
            ABBR = RNA_ABBRS
            BB_LEN = 12
        three_letter = ABBR[pdb_dict[id]["residue_names"][0]]
        residue = PDB.Residue.Residue((" ", id, " "), three_letter, id)
        for n in range(len(pdb_dict[id]["residue_names"])):
            if bb_only and n > BB_LEN - 1:
                break
            if n < BB_LEN:
                atom_name = BB_ATOMS[n]
            else:
                # side-chains
                try:
                    atom_name = SC_ATOMS[three_letter][n - BB_LEN]
                except:
                    atom_name = "X"
                    print(f"Residue {id} had missing atom and rename to {atom_name}")
            atom_coord = pdb_dict[id]["coordinates"][n][0]
            atom = Atom.Atom(
                name=atom_name,
                coord=atom_coord,
                bfactor=0,
                occupancy=1.0,
                altloc=" ",
                fullname=atom_name,
                serial_number=counter,
                element=atom_name[0],
            )

            residue.add(atom)
            counter += 1

        # Add the residue to the chain
        chain.add(residue)

    # Add the chain to the model
    model.add(chain)

    # Add the model to the structure
    structure.add(model)

    if pdb_file_path is not None:
        structure_to_pdb(structure, pdb_file_path)

    return structure


def to_pdb_dict(
    coords: List[np.ndarray],
    continuous_res_ids: List[int],
    residue_names: List[str],
    atom_names: List[str],
    res_types: List[str],
):
    counter = 0
    idx = 1
    coordinates = []
    a_names = []
    res_names = []
    res_type = []
    pdb_dict = {}
    for residue_id in continuous_res_ids:
        if idx == residue_id:
            coordinates.append(coords[counter])
            a_names.append(atom_names[counter])
            res_names.append(residue_names[counter])
            res_type.append(res_types[counter])
        else:
            pdb_dict[idx] = {
                "coordinates": coordinates,
                "atom_names": a_names,
                "residue_names": res_names,
                "res_types": res_type,
            }
            idx += 1
            coordinates = []
            res_names = []
            a_names = []
            res_type = []
            idx = residue_id
            coordinates.append(coords[counter])
            a_names.append(atom_names[counter])
            res_names.append(residue_names[counter])
            res_type.append(res_types[counter])
        counter += 1

    return pdb_dict
