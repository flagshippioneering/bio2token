# bio2token
This repo is for testing of Bio2Token models. 
To load an example pdb from the examples/ folder run `./run_bio2token.py --tokenizer <tokenizer>`
Available tokenizer with checkpoints:
mol2token, protein2token, rna2token, bio2token

To run examples mentioned in the manuscript:
To auto-reconstruct a small rna:
`./run_bio2token --pdb 1rna --tokenizer rna2token`
To auto-reconstruct the biggest RNA chain from the RNA3DB test set:
`./run_bio2token --pdb 8toc-pdb-bundle1 --tokenizer rna2token --chains [R]`
To reconstruct a protein-RNA complex:
`./run_bio2token --pdb 3wbm --tokenizer bio2token`