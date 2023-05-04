import pickle

import numpy as np
import pickle as pkl
from collections import defaultdict
import random
# from sklearn.metrics.pairwise import euclidean_distances
import argparse
import datetime
import json
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

timestamp = 202211151301
model_type = 'EGR_generator'
initial_ranker = 'lambdaMART'
batch_size = 64
lr = 0.0001
l2_reg = 9e-05
hidden_size = 64
eb_dim = 16
keep_prob = 0.8
acc_prefer = 1.0
data_set_name = 'ad'
max_len = 10
controllable = False
with_evaluator_metrics = False

save_dir = 'model/logs_{}/{}/'.format(data_set_name, max_len)
model_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.monitor2.pkl'.format(timestamp, initial_ranker, model_type, batch_size, lr,
                                                                 l2_reg,
                                                                 hidden_size, eb_dim, keep_prob,
                                                                 acc_prefer if not controllable else 'controllable')

#1234
# model_name, model_type = '202301112327_lambdaMART_CMR_generator_16_0.0001_0.0001_64_16_0.8_controllable.monitor2.pkl', \
#                          'CMR_generator'  # 26179
# model_name, model_type = '202301122214_lambdaMART_Seq2Slate_16_0.0001_0.0001_64_16_0.8_controllable.monitor2.pkl', \
#                          'Seq2Slate'  # 26179
# model_name, model_type = '202301161042_lambdaMART_PRM_32_5e-05_0.0001_64_16_0.8_controllable.monitor2.pkl', \
#                          'PRM'  # 26179
# model_name, model_type = '202301132321_lambdaMART_EGR_generator_16_0.0001_0.0001_64_16_0.8_controllable.monitor2.pkl', \
#                          'EGR_generator'  # 26179
#2234
# model_name, model_type = '202301161705_lambdaMART_PRM_32_5e-05_0.0001_64_16_0.8_controllable.monitor2.pkl', \
#                          'PRM'  # 26179
# model_name, model_type = '202301160955_lambdaMART_CMR_generator_16_0.0001_0.0001_64_16_0.8_controllable.monitor2.pkl', \
#                          'CMR_generator'  # 26179
# model_name, model_type = '202301162239_lambdaMART_EGR_generator_16_0.0001_0.0001_64_16_0.8_controllable.monitor2.pkl', \
#                          'EGR_generator'  # 26179
# model_name, model_type = '202301162023_lambdaMART_Seq2Slate_16_0.0001_0.0001_64_16_0.8_controllable.monitor2.pkl', \
#                          'Seq2Slate'  # 26179
# model_name, model_type = '202301171220_lambdaMART_miDNN_64_0.0001_9e-05_64_16_0.8_controllable.monitor2.pkl', \
#                          'miDNN'  # 26179
#3234
# model_name, model_type = '202301181615_lambdaMART_CMR_generator_16_0.0001_0.0001_64_16_0.8_controllable.monitor2.pkl', \
#                          'CMR_generator'  # 26179
# model_name, model_type = '202301181616_lambdaMART_EGR_generator_16_0.0001_0.0001_64_16_0.8_controllable.monitor2.pkl', \
#                          'EGR_generator'  # 26179
# model_name, model_type = '202301181616_lambdaMART_Seq2Slate_16_0.0001_0.0001_64_16_0.8_controllable.monitor2.pkl', \
#                          'Seq2Slate'  # 26179
model_name, model_type = '202303282203_lambdaMART_MTS_generator_16_0.0001_0.0001_64_16_0.8_1.monitor2.pkl', \
                         'MTS_generator'  # 43xxx
f = open(save_dir + model_name, 'rb')
f.flush()
data = pickle.load(f)

def process_data(name, pre_num):
    _m = max(map(lambda a: a[pre_num], data[name][1:]))
    return _m


def process_max_data(name, pre_num, idx):
    print("max_{}@{}".format(name, pre_num))
    _m = max(map(lambda a: a[idx], data[name][1:]))
    print(_m)
    _idx = list(map(lambda a: a[idx], data[name][1:])).index(_m)
    print(data['map_l'][1 + _idx])
    print(data['ndcg_l'][1 + _idx])
    print(data['ilad_l'][1 + _idx])
    print(data['err_ia_l'][1 + _idx])


def process_with_controllable():
    for i, j in enumerate([1, 3, 5, 10]):
        # for name in ['map_l', 'ndcg_l', 'ilad_l', 'err_ia_l']:
        #     print(data[name][0][i], end=', ')
        # print('')
        # for name in ['map_l', 'ndcg_l', 'ilad_l', 'err_ia_l']:
        #     print(process_data(name, i), end=', ')
        map_l, ndcg_l, ilad_l, err_ia_l = [], [], [], []
        for k in ([0, 1, 2]):
            map_l.append(list(map(lambda a: a[k][i], data['map_l'])))
            ndcg_l.append(list(map(lambda a: a[k][i], data['ndcg_l'])))
            ilad_l.append(list(map(lambda a: a[k][i], data['ilad_l'])))
            err_ia_l.append(list(map(lambda a: a[k][i], data['err_ia_l'])))
            print(max(map_l[-1]), max(ndcg_l[-1]), max(ilad_l[-1]), max(err_ia_l[-1]))
            map_l = [gaussian_filter1d(map, sigma=5) for map in map_l]
            ndcg_l = [gaussian_filter1d(ndcg, sigma=5) for ndcg in ndcg_l]
            ilad_l = [gaussian_filter1d(ilad, sigma=5) for ilad in ilad_l]
            err_ia_l = [gaussian_filter1d(err_ia, sigma=5) for err_ia in err_ia_l]
        x = [i / 5 for i in range(len(map_l[0]))]
        plt.subplot(2, 2, 1)
        plt.plot(x, map_l[0], 'r-', label='0')
        plt.plot(x, map_l[1], 'g-', label='0.5')
        plt.plot(x, map_l[2], 'b-', label='1')
        plt.xlabel('epoch')
        plt.ylabel('map')
        plt.legend()
        plt.subplot(2, 2, 2)
        plt.plot(x, ndcg_l[0], 'r-', label='0')
        plt.plot(x, ndcg_l[1], 'g-', label='0.5')
        plt.plot(x, ndcg_l[2], 'b-', label='1')
        plt.xlabel('epoch')
        plt.ylabel('ndcg')
        plt.legend()
        plt.subplot(2, 2, 3)
        plt.plot(x, ilad_l[0], 'r-', label='0')
        plt.plot(x, ilad_l[1], 'g-', label='0.5')
        plt.plot(x, ilad_l[2], 'b-', label='1')
        plt.xlabel('epoch')
        plt.ylabel('ilad')
        plt.legend()
        plt.subplot(2, 2, 4)
        plt.plot(x, err_ia_l[0], 'r-', label='0')
        plt.plot(x, err_ia_l[1], 'g-', label='0.5')
        plt.plot(x, err_ia_l[2], 'b-', label='1')
        plt.xlabel('epoch')
        plt.ylabel('err_ia')
        plt.legend()
        plt.suptitle("{}_{}_@{}".format(model_type, 'controllable', j))
        plt.legend()
        plt.show()


def process_without_controllable():
    train_loss = list(map(lambda a: a, data['train_loss']))
    vali_loss = list(map(lambda a: a, data['vali_loss']))
    x = [i / 5 for i in range(len(train_loss)-1)]
    plt.subplot(1, 2, 1)
    plt.plot(x, train_loss[1:], 'r-')
    plt.xlabel('epoch')
    plt.ylabel('train_loss')
    plt.subplot(1, 2, 2)
    plt.plot(x, vali_loss[1:], 'g-')
    plt.xlabel('epoch')
    plt.ylabel('vali_loss')
    plt.legend()
    plt.show()
    for i, j in enumerate([1, 3, 5, 10]):
        for name in ['map_l', 'ndcg_l', 'ilad_l', 'err_ia_l']:
            print(data[name][0][i], end=', ')
        print('')
        for name in ['map_l', 'ndcg_l', 'ilad_l', 'err_ia_l']:
            print(process_data(name, i), end=', ')
        map_l = list(map(lambda a: a[i], data['map_l']))
        ndcg_l = list(map(lambda a: a[i], data['ndcg_l']))
        ilad_l = list(map(lambda a: a[i], data['ilad_l']))
        err_ia_l = list(map(lambda a: a[i], data['err_ia_l']))

        map_l = gaussian_filter1d(map_l, sigma=5)
        ndcg_l = gaussian_filter1d(ndcg_l, sigma=5)
        ilad_l = gaussian_filter1d(ilad_l, sigma=5)
        err_ia_l = gaussian_filter1d(err_ia_l, sigma=5)
        x = [i / 5 for i in range(len(map_l))]
        plt.subplot(2, 2, 1)
        plt.plot(x, map_l, 'r-')
        plt.xlabel('epoch')
        plt.ylabel('map')
        plt.subplot(2, 2, 2)
        plt.plot(x, ndcg_l, 'g-')
        plt.xlabel('epoch')
        plt.ylabel('ndcg')
        plt.subplot(2, 2, 3)
        plt.plot(x, ilad_l, 'b-')
        plt.xlabel('epoch')
        plt.ylabel('ilad')
        plt.subplot(2, 2, 4)
        plt.plot(x, err_ia_l, 'y-')
        plt.xlabel('epoch')
        plt.ylabel('err_ia')
        plt.suptitle("{}_{}_@{}".format(model_type, acc_prefer, j))
        plt.legend()
        plt.show()

    process_max_data("map_l", 5, 2)
    process_max_data("map_l", 10, 3)
    process_max_data("ndcg_l", 5, 2)
    process_max_data("ndcg_l", 10, 3)
    process_max_data("ilad_l", 3, 1)
    process_max_data("ilad_l", 5, 2)
    process_max_data("err_ia_l", 3, 1)
    process_max_data("err_ia_l", 5, 2)
    process_max_data("err_ia_l", 10, 3)

def process_with_evaluator_metrics():
    train_loss = list(map(lambda a: a, data['train_loss']))
    vali_loss = list(map(lambda a: a, data['vali_loss']))
    x = [i / 5 for i in range(len(train_loss)-1)]
    plt.subplot(1, 2, 1)
    plt.plot(x, train_loss[1:], 'r-')
    plt.xlabel('epoch')
    plt.ylabel('train_loss')
    plt.subplot(1, 2, 2)
    plt.plot(x, vali_loss[1:], 'g-')
    plt.xlabel('epoch')
    plt.ylabel('vali_loss')
    plt.legend()
    plt.show()
    for i, j in enumerate([1, 3, 5, 10]):
        for name in ['map_l', 'ndcg_l', 'eva_sum', 'eva_ave']:
            print(data[name][0][i], end=', ')
        print('')
        for name in ['map_l', 'ndcg_l', 'eva_sum', 'eva_ave']:
            print(process_data(name, i), end=', ')
        map_l = list(map(lambda a: a[i], data['map_l']))
        ndcg_l = list(map(lambda a: a[i], data['ndcg_l']))
        sum_l = list(map(lambda a: a[i], data['eva_sum']))
        ave_l = list(map(lambda a: a[i], data['eva_ave']))

        map_l = gaussian_filter1d(map_l, sigma=5)
        ndcg_l = gaussian_filter1d(ndcg_l, sigma=5)
        sum_l = gaussian_filter1d(sum_l, sigma=5)
        ave_l = gaussian_filter1d(ave_l, sigma=5)
        x = [i / 5 for i in range(len(map_l))]
        plt.subplot(2, 2, 1)
        plt.plot(x, map_l, 'r-')
        plt.xlabel('epoch')
        plt.ylabel('map')
        plt.subplot(2, 2, 2)
        plt.plot(x, ndcg_l, 'g-')
        plt.xlabel('epoch')
        plt.ylabel('ndcg')
        plt.subplot(2, 2, 3)
        plt.plot(x, sum_l, 'b-')
        plt.xlabel('epoch')
        plt.ylabel('eva_sum')
        plt.subplot(2, 2, 4)
        plt.plot(x, ave_l, 'y-')
        plt.xlabel('epoch')
        plt.ylabel('eva_ave')
        plt.suptitle("{}_{}_@{}".format(model_type, acc_prefer, j))
        plt.legend()
        plt.show()

    process_max_data("map_l", 5, 2)
    process_max_data("map_l", 10, 3)
    process_max_data("ndcg_l", 5, 2)
    process_max_data("ndcg_l", 10, 3)
    process_max_data("eva_sum", 3, 1)
    process_max_data("eva_sum", 5, 2)
    process_max_data("eva_ave", 3, 1)
    process_max_data("eva_ave", 5, 2)
    process_max_data("eva_ave", 10, 3)


if model_name.endswith("controllable.monitor2.pkl"):
    process_with_controllable()
elif with_evaluator_metrics:
    process_with_evaluator_metrics()
else:
    process_without_controllable()


# model_name_2 = '202303051741_lambdaMART_LAST_generator_16_0.0001_0.0001_64_16_0.8_1.monitor.pkl'
# f2 = open(save_dir+model_name_2, 'rb')
# data2=pickle.load(f2)
# data['eva_sum'] = [data2['eva_sum'][2*i+1] for i in range(len(data2['eva_sum'])//2)]
# data['eva_ave'] = [data2['eva_ave'][2*i+1] for i in range(len(data2['eva_ave'])//2)]
# pkl.dump(data, open(save_dir+model_name, 'wb'))