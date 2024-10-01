# bio2token
This repo is for testing of Bio2Token models. 
To load an example pdb from the examples/ folder run `./run_bio2token.py --pdb <xxxx> --tokenizer <tokenizer>` \n
Available tokenizer with checkpoints: \n
mol2token, protein2token, rna2token, bio2token

To run examples mentioned in the manuscript:\n
To auto-reconstruct a small rna:\n
```./run_bio2token --pdb 1rna --tokenizer rna2token```\n
To auto-reconstruct the biggest RNA chain from the RNA3DB test set:
`./run_bio2token --pdb 8toc-pdb-bundle1 --tokenizer rna2token --chains [R]`
To reconstruct a protein-RNA complex:
`./run_bio2token --pdb 3wbm --tokenizer bio2token`