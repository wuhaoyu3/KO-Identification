# KO-Identification

## Requirements
- Python 3.6
- Dependencies: See requirements.txt

## Getting Started
```sh
# get ProtT5 embedding (input format: fasta, output format: h5)
python prott5_seq2embedding.py seq_file.fa embedding_file.h5

# use the trained classifier to make predictions (output format: csv)
python prediction.py model/mlp_pipe.pt embedding_file.h5 result_file.csv

# cluster (result file format: csv)
python cluster.py embedding_file.h5 reference.h5 result_file.csv

# train and test the classifier (The paths to the model and dataset can be modified in the python file)
python cls/mlp_train.py
python cls/mlp_test.py
```
## Models
If you want to apply the model directly, you should use `mlp_pipe.pt`.

```
└── Model
    ├── att_cls.pt          # attention model
    ├── lstm_cls.pt         # LSTM model
    ├── mlp_cls.pt          # MLP model (used for classifier testing)
    └── mlp_pipe.pt         # MLP model (used for testing the entire pipeline)
```

## Data availability
Publicly available datasets were analyzed in our paper. These datasets were collected from the [KEGG](https://www.kegg.jp/) database, the [PDB](https://www.rcsb.org/) database, and the [AFDB](https://alphafold.ebi.ac.uk/) database.
