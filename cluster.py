import numpy as np
import h5py
import pandas as pd
from pandas import read_excel, DataFrame
from bio_embeddings.extract import pairwise_distance_matrix_from_embeddings_and_annotations
from sklearn.metrics.pairwise import cosine_similarity
import copy
import os
import sys

ref_list = 'ref_list'
target_list = 'pair_list'
target_result = 'pair_result'
result_file = 'result'

target_file = sys.argv[1]
ref_file = sys.argv[2]
result_dir = sys.argv[3]

matrix, list1, list2 = pairwise_distance_matrix_from_embeddings_and_annotations(target_file, ref_file,n_jobs=8)

print(len(matrix))
print(len(matrix[0]))

# save
a = np.array(matrix)
np.save(os.path.join(result_dir, target_result+'.npy'), a)
a = np.array(list1)
np.save(os.path.join(result_dir, target_list+'.npy'), a)
a = np.array(list2)
np.save(os.path.join(result_dir, ref_list+'.npy'), a)

# get best match
with open(os.path.join(result_dir, result_file+'.csv'), 'w') as f:
    for i in range(len(matrix)):
        f.write("\"" + list1[i]+"\",")
        m = list(matrix[i])
        number = min(m)
        index = m.index(number)
        f.write(str(number)+",")
        f.write("\""+list2[index].split()[0]+'\",\"'+list2[index].split()[1]+'\",')
        print("\r{}".format(i).ljust(20), end='', flush=True)
        f.write("\n")
