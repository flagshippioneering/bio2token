#!/usr/bin/env python3
from typing import *
import torch
import argparse
import os
import logging

from bio2token.models.fsq_ae import FSQ_AE


from bio2token.utils.utils import *
from bio2token.utils.pdb import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="bio2token", help="Tokenizer to use")
    parser.add_argument("--pdb", type=str, default="6n64", help="PDB file to parse")
    parser.add_argument("--chains", type=str, default=None, help="Chains to parse from pdb, e.g. [A,C]")
    parser.add_argument(
        "--seq_type", type=str, default="AA", help="Sequence type, RNA or AA, only needed for TM-score calculation"
    )
    args = parser.parse_args()

    if args.tokenizer == "mol2token":
        checkpoint = "./checkpoints/mol2token.ckpt"
    if args.tokenizer == "protein2token":
        checkpoint = "./checkpoints/protein2token.ckpt"
    if args.tokenizer == "bio2token":
        checkpoint = "./checkpoints/bio2token.ckpt"
    if args.tokenizer == "rna2token":
        checkpoint = "./checkpoints/rna2token.ckpt"
    config_model = load_config("./configs/tokenizer.yaml")

    # Instantiate and move the model to GPU
    model = FSQ_AE(config_model).cuda()

    state_dict = torch.load(checkpoint, map_location="cuda:0")["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace("model.", "")] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Print the number of parameters
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info(f"Number of parameters: {count_parameters(model)}")
    biomolecule = pdb_2_dict(os.path.join("./examples/ground_truth", args.pdb + ".pdb"), chains=args.chains)
    batch = pdb_to_batch(config_model, biomolecule)
    batch["seq_type"] = args.seq_type

    logging.info(f"PDB file: {args.pdb}")
    logging.info(f"Chains: {args.chains}")
    logging.info(f"Number of residues: {len(biomolecule['seq'])}")
    logging.info(f"Number of atoms: {len(biomolecule['atom_names'])}")

    with torch.no_grad():
        out = model.step(batch, mode="inference")

    logging.info(f"RMSD error: {out['rmsd'].item()}")
    logging.info(f"TM-score: {out['tm'].item()}")
    # Cut out zeros in batch
    gt = out["coords_gt"][:, : biomolecule["atom_length"], :].squeeze(0).cpu().numpy()
    gt = np.split(gt, gt.shape[0])
    recon = out["coords_pred_kabsch_all"][:, : biomolecule["atom_length"], :].squeeze(0).cpu().numpy()
    recon = np.split(recon, recon.shape[0])

    pdb_dict_gt = to_pdb_dict(
        coords=gt,
        atom_names=biomolecule["atom_names"],
        continuous_res_ids=biomolecule["continuous_res_ids"],
        residue_names=biomolecule["res_names"],
        res_types=biomolecule["res_types"],
    )
    pdb_dict_recon = to_pdb_dict(
        coords=recon,
        atom_names=biomolecule["atom_names"],
        continuous_res_ids=biomolecule["continuous_res_ids"],
        residue_names=biomolecule["res_names"],
        res_types=biomolecule["res_types"],
    )
    # print(pdb_dict_recon)
    pdb_dict_to_file(
        pdb_dict_gt,
        pdb_file_path=os.path.join("./examples/recon", f"{args.pdb}_{args.tokenizer}_gt.pdb"),
    )

    pdb_dict_to_file(
        pdb_dict_recon,
        pdb_file_path=os.path.join("./examples/recon", f"{args.pdb}_{args.tokenizer}_recon.pdb"),
    )


if __name__ == "__main__":
    main()
