#!/usr/bin/env python3
from typing import *
import torch
from torch.utils.data import DataLoader
from loguru import logger
import argparse

from bio2token.models.fsq_ae import FSQ_AE
from functools import partial


from bio2token.utils.utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="protein2token")
    args = parser.parse_args()

    if args.tokenizer == "protein2token":
        checkpoint = "/home/pi-user/bio2token/checkpoints/protein2token.ckpt"
        config_model = load_config("/home/pi-user/bio2token/configs/protein2token.yaml")

    # Instantiate and move the model to GPU
    model = FSQ_AE(config_model).cuda()

    state_dict = torch.load(checkpoint)["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace("model.", "")] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Print the number of parameters
    print(f"Number of parameters: {count_parameters(model)}")
    protein = pdb_2_dict("/home/pi-user/bio2token/examples/6vn1.pdb", chains=["L"])
    batch = pdb_to_batch(config_model, protein)
    batch["seq_type"] = "protein"
    with torch.no_grad():
        out = model.step(batch, mode="inference")

    print("RMSD error: ", out["rmsd"])
    print("TM-score: ", out["tm"])


if __name__ == "__main__":
    main()
