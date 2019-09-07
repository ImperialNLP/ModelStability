import pickle
from scipy.stats import entropy
from itertools import combinations
import numpy as np
import sys

sys.path.append('/home/ubuntu/')
sys.path.append('/data/rishabh/')
sys.path.append('/data/rishabh/Transparency')
from Transparency.Trainers.DatasetBC import filterbylength, sortbylength
from sklearn.calibration import calibration_curve


def print_map(map):
    str_map = ""
    for l in map:
        str_map += str(l) + ','
    print(str_map)


def get_all_outputs(filename):
    all_outputs = pickle.load(
        open(filename, 'rb'))
    return all_outputs


def get_high_entropy_cases():
    num_models = 10
    all_outputs = get_all_outputs(filename)[:num_models]
    new_all_outputs = get_all_outputs(filename_new)[:num_models]
    combins = list(combinations(range(num_models), 2))
    max_entropy_cases = []
    good_cases = []
    accuracy = [0, 0]
    best_improvement = 0
    for test_num in range(len(all_outputs[0][0])):
        avg_entropy = 0
        avg_entropy_new = 0
        preds, preds_new = [], []
        for comb in combins:
            model_i, model_j = comb[0], comb[1]
            # print(len(all_outputs[model_i][1]), test_num)
            # print(all_outputs[model_i][1], test_num)
            # import ipdb; ipdb.set_trace()
            atn_vec_i, atn_vec_j = all_outputs[model_i][1][test_num], \
                                   all_outputs[model_j][1][test_num]
            atn_vec_new_i, atn_vec_new_j = new_all_outputs[model_i][1][
                                               test_num], \
                                           new_all_outputs[model_j][1][test_num]
            entropy_i = entropy(atn_vec_i, atn_vec_j)
            entropy_i_new = entropy(atn_vec_new_i, atn_vec_new_j)
            avg_entropy += entropy_i
            avg_entropy_new += entropy_i_new
            preds.append(all_outputs[model_i][0][test_num])
            preds_new.append(new_all_outputs[model_i][0][test_num])
            if entropy_i > 1.0:
                max_entropy_cases.append([test_num, model_i, model_j,
                                          all_outputs[model_i][0][test_num],
                                          all_outputs[model_j][0][test_num],
                                          atn_vec_i,
                                          atn_vec_j])
        avg_entropy /= len(combins)
        avg_entropy_new /= len(combins)
        if np.sign(np.mean(preds_new) - 0.5) == np.sign(yt[test_num] - 0.5):
            accuracy[0] += 1
        else:
            accuracy[1] += 1

        if avg_entropy > 0.55 and avg_entropy_new < (1-0.1) * avg_entropy and avg_entropy_new > (1-0.99) * avg_entropy:
            if avg_entropy_new < avg_entropy:
                impr = 1 - avg_entropy_new / avg_entropy
                if impr > best_improvement:
                    best_improvement = impr
            str1 = "OLD: ", avg_entropy, np.std(preds), np.mean(preds)
            str2 = "NEW:", avg_entropy_new, np.std(preds_new), np.mean(
                preds_new)

            good_cases.append((test_num, str1, str2))
    print(best_improvement)
    # exit(0)

    # for max_entropy_case in max_entropy_cases:
    #     if max_entropy_case[3] > 0.1 or max_entropy_case[4] > 0.1:
    #         continue
    #     test_case = Xt[max_entropy_case[0]]
    #     test_case_text = []
    #     for idx in test_case:
    #         test_case_text.append(vec.idx2word[idx])
    #     vec_1_map, vec_2_map = {}, {}
    #     for i in range(len(test_case_text)):
    #         word = test_case_text[i]
    #         vec_1_map[word] = max_entropy_case[5][i] // 0.01 / 100
    #         vec_2_map[word] = max_entropy_case[6][i] // 0.01 / 100
    #     print(max_entropy_case[0], 'test case')
    #     print(' '.join(test_case_text))
    #     print(max_entropy_case[3],
    #           max_entropy_case[4])
    #     sorted_vec1 = sorted(vec_1_map.items(), key=lambda kv: kv[1],
    #                          reverse=True)
    #     sorted_vec2 = sorted(vec_2_map.items(), key=lambda kv: kv[1],
    #                          reverse=True)
    #     print(sorted_vec1)
    #     print(sorted_vec2, '\n\n')
    print(accuracy)
    # exit(1)
    for good_case, str1, str2 in good_cases:
        test_case = Xt[good_case]
        prediction = yt[good_case]
        if prediction == 0:
            continue
        test_case_text = []
        for idx in test_case:
            test_case_text.append(vec.idx2word[idx])
        if len(test_case_text) > 40:
            continue
        vec_og_maps, vec_new_maps = [], []
        vec_atns, vec_new_atns = [], []
        for model_num in range(num_models):
            vec_og = {}
            vec_new = {}
            vec_atn_og = all_outputs[model_num][1][good_case]
            vec_atn_new = new_all_outputs[model_num][1][good_case]

            for i in range(len(test_case_text)):
                word = test_case_text[i]
                vec_og[word] = vec_atn_og[i] // 0.01 / 100
                vec_new[word] = vec_atn_new[i] // 0.01 / 100
            sorted_vec_og = sorted(vec_og.items(), key=lambda kv: kv[1],
                                   reverse=True)
            sorted_vec_new = sorted(vec_new.items(), key=lambda kv: kv[1],
                                    reverse=True)
            vec_og_maps.append(sorted_vec_og)
            vec_new_maps.append(sorted_vec_new)

            vec_atns.append(vec_atn_og)
            vec_new_atns.append(vec_atn_new)

        print(' '.join(test_case_text))
        print("Correct pred", prediction)
        print(str1)
        print(str2)
        for a in vec_og_maps:
            print_map(a)
        print('\n')
        for a in vec_new_maps:
            print_map(a)
        print('\n\n')
        for a in vec_atns:
            print_map(a)
        print('\n')
        for a in vec_new_atns:
            print_map(a)
        print('\n\n\n\n')


def do_analysis_on_test(test_num):
    num_models = 10
    og_all_outputs = get_all_outputs(
        "../final-pkl-files-old/stability-outputs-[0,0,0,0][1,1024,2**30,43,789,1537,7771,2**18,99999,13]tanhsstlstm1.pkl")[
                     :num_models]
    stable_all_outputs = get_all_outputs(
        "../final-pkl-files-old/stability-outputs-[1,1,1,0][1,1024,2**30,43,789,1537,7771,2**18,99999,13]tanhsstlstm1.pkl")[
                         :num_models]
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
        sorted_vec1 = sorted(vec_1_map.items(), key=lambda kv: kv[1],
                             reverse=True)
        sorted_vec2 = sorted(vec_2_map.items(), key=lambda kv: kv[1],
                             reverse=True)

        print('og', sorted_vec1)
        print('new', sorted_vec2, '\n\n')


filename = '../all-pkl-files/stability-outputs-[0,0,0,0][1,1024,2**30,43,789,1537,7771,2**18,99999,13]tanhimdblstm1.pkl'
filename_new = '../all-pkl-files/stability-outputs-[1,1,1,1][1,1024,2**30,43,789,1537,7771,2**18,99999,13]tanhimdblstm1.pkl'

vec = pickle.load(open('../preprocess/IMDB/vec_imdb.p', 'rb'))
Xt, yt = vec.seq_text['test'], vec.label['test']
# Xt, yt = filterbylength(Xt, yt, min_length=5, max_length=100)
Xt, yt = filterbylength(Xt, yt, min_length=6)
# Xt, yt = filterbylength(Xt, yt)
Xt, yt = sortbylength(Xt, yt)
get_high_entropy_cases()
# do_analysis_on_test(1553)
