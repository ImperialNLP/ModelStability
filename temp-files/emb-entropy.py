from itertools import combinations
import pickle
from scipy.stats import entropy
import numpy as np

dataset = 'diab'
model_info = 'lstm+tanh'

dist_mtxs = pickle.load(
    open(dataset + model_info + '-distance-matrices.pkl', 'rb'))

num_models = 10
dist_mtxs = dist_mtxs[:num_models]

combins = list(combinations(range(num_models), 2))

all_entrs = []
for word_num in range(len(dist_mtxs[0])):
    entrs = []
    for comb in combins:
        model_i, model_j = comb[0], comb[1]
        entr = entropy(dist_mtxs[model_i][word_num],
                       dist_mtxs[model_j][word_num])
        entrs.append(entr)
    avg_entr = np.mean(entrs)
    all_entrs.append(avg_entr)

import pickle

file_name = dataset + model_info + "-entropy.pkl"
pkl_file = open(file_name, 'wb')
pickle.dump(all_entrs, pkl_file)
pkl_file.close()
exit(0)
