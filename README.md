# bio2token
This repo is for testing of Bio2Token models. <br>
To load an example pdb from the `examples/` folder (or add your own) run <br>
```.scripts/run_bio2token.py --pdb <XXXX> ``` <br>
Other optional args are: <br>
- `--tokenizer`: bio2token(default), protein2token, mol2token, rna2token
- `--chains` : list of chains to parse from pdb, e.g. `[A]`, default is all

Available tokenizer with checkpoints: <br>
- mol2token
- protein2token
- rna2token
- bio2token

All reconstructed pdbs will be written into the folder `examples/recon`: <br>
- <pdb>_<tokenizer>_recon.pdb : auto-reconstructed pdb
- <pdb>_<tokenizer>_gt.pdb : original ground truth structure (or chains that were selected in the input), centered and written to pdb 

## Example calls
To run examples mentioned in the manuscript: <br>
To auto-reconstruct a small rna:<br>
```./run_bio2token --pdb 1rna --tokenizer rna2token```<br>
To auto-reconstruct the biggest RNA chain from the RNA3DB test set: <br>
```./run_bio2token --pdb 8toc-pdb-bundle1 --tokenizer rna2token --chains [R]``` <br>
To reconstruct a protein-RNA complex: <br>
```./run_bio2token --pdb 3wbm --tokenizer bio2token``` <br>
