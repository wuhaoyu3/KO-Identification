import numpy as np
import h5py
from typing import List, Tuple
import pandas as pd
from sklearn.metrics import pairwise_distances as _pairwise_distances
import sys

query_embeddings_path = sys.argv[1]
reference_embeddings_path = sys.argv[2]
result_file_path = sys.argv[3]

references: List[str]
reference_embeddings = list()

with h5py.File(reference_embeddings_path, 'r') as reference_embeddings_file:

    references = list(reference_embeddings_file.keys())

    length = len(references)
    cur = 0
    for refereince_identifier in references:
        reference_embeddings.append(np.array(reference_embeddings_file[refereince_identifier]))
        cur += 1
        print(f"\rreference: {cur}/{length}".ljust(20), end='', flush=True)

queries: List[str]
query_embeddings = list()

with h5py.File(query_embeddings_path, 'r') as query_embeddings_file:

    queries = list(query_embeddings_file.keys())

    length = len(queries)
    cur = 0
    for query_identifier in queries:
        query_embeddings.append(np.array(query_embeddings_file[query_identifier]))
        cur += 1
        print(f"\rquery: {cur}/{length}".ljust(20), end='', flush=True)


pairwise_distances = _pairwise_distances(
        query_embeddings,
        reference_embeddings,
        metric='euclidean',
        n_jobs=-1
)

# get best match
with open(result_file_path, 'w') as f:
    for i in range(len(matrix)):
        f.write("\"" + list1[i]+"\",")
        min_index = np.where(matrix[i] == np.amin(matrix[i]))[0]
        for j in range(len(min_index)):
            f.write(str(matrix[i][j])+",")
            f.write("\""+list2[j].split()[0]+'\",\"'+list2[j].split()[1]+'\",')
        print("\r{}".format(i).ljust(20), end='', flush=True)
        f.write("\n")
