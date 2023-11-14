# KO-Identification

## Requirements
- Python 3.9
- Dependencies: See requirements.txt

## Getting Started
```sh
# get ProtT5 embedding
python prott5_seq2embedding.py seq_file.fa embedding_file.h5

# use the trained classifier to make predictions
python prediction.py model/mlp_pipe.pt embedding_file.h5 result_file

# cluster
python cluster.py embedding_file.h5 reference.h5 result_dir

# train and test the classifier (The paths to the model and dataset can be modified in the python file)
python cls/mlp_train.py
python cls/mlp_test.py
```

## Data availability
Publicly available datasets were analyzed in our paper. These datasets were collected from the [KEGG](https://www.kegg.jp/) database, the [PDB](https://www.rcsb.org/) database, and the [AFDB](https://alphafold.ebi.ac.uk/) database.
