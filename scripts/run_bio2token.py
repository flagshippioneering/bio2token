#!/usr/bin/env python3
from typing import *
import torch
import argparse
import os

from bio2token.models.fsq_ae import FSQ_AE


from bio2token.utils.utils import *
from bio2token.utils.pdb import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="bio2token")
    parser.add_argument("--pdb_file", type=str, default="6n64.pdb")
    parser.add_argument("--res_type", type=str, default="AA")
    args = parser.parse_args()

    if args.tokenizer == "mol2token":
        checkpoint = "/home/pi-user/bio2token/checkpoints/mol2token.ckpt"
        config_model = load_config("/home/pi-user/bio2token/configs/tokenizer.yaml")
    if args.tokenizer == "protein2token":
        checkpoint = "/home/pi-user/bio2token/checkpoints/protein2token.ckpt"
        config_model = load_config("/home/pi-user/bio2token/configs/tokenizer.yaml")
    if args.tokenizer == "bio2token":
        checkpoint = "/home/pi-user/bio2token/checkpoints/bio2token.ckpt"
        config_model = load_config("/home/pi-user/bio2token/configs/tokenizer.yaml")

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
    print(f"Number of parameters: {count_parameters(model)}")
    biomolecule = pdb_2_dict(
        os.path.join("/home/pi-user/bio2token/examples/ground_truth", args.pdb_file), res_type=args.res_type, chains=["A"]
    )
    batch = pdb_to_batch(config_model, biomolecule)
    batch["seq_type"] = args.res_type
    with torch.no_grad():
        out = model.step(batch, mode="inference")

    print("RMSD error: ", out["rmsd"])
    print("TM-score: ", out["tm"])
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
    )
    pdb_dict_recon = to_pdb_dict(
        coords=recon,
        atom_names=biomolecule["atom_names"],
        continuous_res_ids=biomolecule["continuous_res_ids"],
        residue_names=biomolecule["res_names"],
    )
    # print(pdb_dict_recon)
    pdb_dict_to_file(
        pdb_dict_gt,
        pdb_file_path=os.path.join("/home/pi-user/bio2token/examples/recon", f"{args.pdb_file}_gt.pdb"),
        res_type=args.res_type,
    )

    pdb_dict_to_file(
        pdb_dict_recon,
        pdb_file_path=os.path.join("/home/pi-user/bio2token/examples/recon", f"{args.pdb_file}_recon.pdb"),
        res_type=args.res_type,
    )


if __name__ == "__main__":
    main()
