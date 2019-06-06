import pickle
from scipy.stats import entropy
from itertools import combinations
import numpy as np
import sys

sys.path.append('/home/ubuntu/')
sys.path.append('/data/rishabh/')
from Trainers.DatasetBC import filterbylength, sortbylength
from common_code.metrics import calc_metrics_classification
from scripts.atn_weight_stability import get_all_outputs


def get_high_entropy_cases():
    num_models = 10
    all_outputs = get_all_outputs(filename)[:num_models]
    combins = list(combinations(range(num_models), 2))
    max_entropy_cases = []
    for test_num in range(len(all_outputs[0][0])):
        for comb in combins:
            model_i, model_j = comb[0], comb[1]
            atn_vec_i, atn_vec_j = all_outputs[model_i][1][test_num], \
                                   all_outputs[model_j][1][test_num]
            entropy_i = entropy(atn_vec_i, atn_vec_j)
            if entropy_i > 1.0:
                max_entropy_cases.append([test_num, model_i, model_j,
                                          all_outputs[model_i][0][test_num],
                                          all_outputs[model_j][0][test_num],
                                          atn_vec_i,
                                          atn_vec_j])
    for max_entropy_case in max_entropy_cases:
        if max_entropy_case[3] > 0.1 or max_entropy_case[4] > 0.1:
            continue
        test_case = Xt[max_entropy_case[0]]
        test_case_text = []
        for idx in test_case:
            test_case_text.append(vec.idx2word[idx])
        vec_1_map, vec_2_map = {}, {}
        for i in range(len(test_case_text)):
            word = test_case_text[i]
            vec_1_map[word] = max_entropy_case[5][i] // 0.01 / 100
            vec_2_map[word] = max_entropy_case[6][i] // 0.01 / 100
        print(max_entropy_case[0], 'test case')
        print(' '.join(test_case_text))
        print(max_entropy_case[3],
              max_entropy_case[4])
        sorted_vec1 = sorted(vec_1_map.items(), key=lambda kv: kv[1], reverse=True)
        sorted_vec2 = sorted(vec_2_map.items(), key=lambda kv: kv[1], reverse=True)
        print(sorted_vec1)
        print(sorted_vec2, '\n\n')

def do_analysis_on_test(test_num):
    num_models = 10
    og_all_outputs = get_all_outputs("../final-pkl-files-old/stability-outputs-[0,0,0,0][1,1024,2**30,43,789,1537,7771,2**18,99999,13]tanhsstlstm1.pkl")[:num_models]
    stable_all_outputs = get_all_outputs("../final-pkl-files-old/stability-outputs-[1,1,1,0][1,1024,2**30,43,789,1537,7771,2**18,99999,13]tanhsstlstm1.pkl")[:num_models]
    og_atns = []
    new_atns = []
    for model_idx in range(len(og_all_outputs)):
        og_atns.append(og_all_outputs[model_idx][1][test_num])
        new_atns.append(stable_all_outputs[model_idx][1][test_num])

    for i in range(len(new_atns)):
        og_atn_vec = og_atns[i]
        new_atn_vec = new_atns[i]
        test_case = Xt[test_num]
        test_case_text = []
        for idx in test_case:
            test_case_text.append(vec.idx2word[idx])
        vec_1_map, vec_2_map = {}, {}
        for j in range(len(test_case_text)):
            word = test_case_text[j]
            vec_1_map[word] = og_atn_vec[j] // 0.01 / 100
            vec_2_map[word] = new_atn_vec[j] // 0.01 / 100
        sorted_vec1 = sorted(vec_1_map.items(), key=lambda kv: kv[1], reverse=True)
        sorted_vec2 = sorted(vec_2_map.items(), key=lambda kv: kv[1], reverse=True)

        print('og', sorted_vec1)
        print('new', sorted_vec2, '\n\n')

filename = '../final-pkl-files-old/stability-outputs-[0,0,0,0][1,1024,2**30,43,789,1537,7771,2**18,99999,13]tanhsstlstm1.pkl'

vec = pickle.load(open('../preprocess/SST/vec_sst.p', 'rb'))
Xt, yt = vec.seq_text['test'], vec.label['test']
Xt, yt = filterbylength(Xt, yt, min_length=5)
Xt, yt = sortbylength(Xt, yt)
# get_high_entropy_cases()
do_analysis_on_test(1553)