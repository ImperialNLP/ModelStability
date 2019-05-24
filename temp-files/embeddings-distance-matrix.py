import os
import torch
import numpy as np

from scipy.spatial import distance_matrix

dataset = 'sst'
model_info = 'lstm+tanh'

dirname = '../outputs/' + dataset + '/' + model_info
dirs = [d for d in os.listdir(dirname) if
        'enc.th' in os.listdir(os.path.join(dirname, d))]
# print(dirs)

distance_matrices = []
for dir in dirs:
    pth = os.path.join(dirname, dir, 'enc.th')
    enc = torch.load(pth, map_location='cpu')
    enc = enc['embedding.weight'].cpu().numpy()
    print(enc.shape)
    distance_matrices.append(
        distance_matrix(enc, enc))  # Minkowski distance with p-norm=2
import pickle

file_name = dataset + model_info + "-distance-matrices.pkl"
pkl_file = open(file_name, 'wb')
pickle.dump(distance_matrices, pkl_file)
pkl_file.close()
