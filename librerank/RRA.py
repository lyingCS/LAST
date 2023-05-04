import numpy as np
import heapq
import os
# from sklearn.metrics import log_loss, roc_auc_score
import random
import time

from numpy import mean

from librerank.utils import *
from librerank.reranker import *
from librerank.rl_reranker import *
import datetime
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def reranker_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_time_len', default=10, type=int, help='max time length')
    parser.add_argument('--save_dir', type=str, default='./', help='dir that saves logs and model')
    parser.add_argument('--data_dir', type=str, default='./data/ad/', help='data dir')
    parser.add_argument('--model_type', default='PRM',
                        choices=['PRM', 'DLCM', 'SetRank', 'GSF', 'miDNN', 'Seq2Slate', 'EGR_evaluator',
                                 'EGR_generator', 'CMR'],
                        type=str,
                        help='algorithm name, including PRM, DLCM, SetRank, GSF, miDNN, Seq2Slate, EGR_evaluator, EGR_generator')
    parser.add_argument('--data_set_name', default='ad', type=str, help='name of dataset, including ad and prm')
    parser.add_argument('--initial_ranker', default='lambdaMART', choices=['DNN', 'lambdaMART'], type=str,
                        help='name of dataset, including DNN, lambdaMART')
    parser.add_argument('--epoch_num', default=30, type=int, help='epochs of each iteration.')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--rep_num', default=5, type=int, help='samples repeat number')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--l2_reg', default=1e-4, type=float, help='l2 loss scale')
    parser.add_argument('--keep_prob', default=0.8, type=float, help='keep probability')
    parser.add_argument('--eb_dim', default=16, type=int, help='size of embedding')
    parser.add_argument('--hidden_size', default=64, type=int, help='hidden size')
    parser.add_argument('--group_size', default=1, type=int, help='group size for GSF')
    parser.add_argument('--acc_prefer', default=1.0, type=float, help='accuracy_prefer/(accuracy_prefer+diversity)')
    parser.add_argument('--metric_scope', default=[1, 3, 5, 10], type=list, help='the scope of metrics')
    parser.add_argument('--max_norm', default=0, type=float, help='max norm of gradient')
    parser.add_argument('--c_entropy', default=0.001, type=float, help='entropy coefficient in loss')
    # parser.add_argument('--decay_steps', default=3000, type=int, help='learning rate decay steps')
    # parser.add_argument('--decay_rate', default=1.0, type=float, help='learning rate decay rate')
    parser.add_argument('--timestamp', type=str, default=datetime.datetime.now().strftime("%Y%m%d%H%M"))
    parser.add_argument('--evaluator_path', type=str, default='', help='evaluator ckpt dir')
    parser.add_argument('--reload_path', type=str, default='', help='model ckpt dir')
    # parser.add_argument('--setting_path', type=str, default='./config/prm_setting.json', help='setting dir')
    parser.add_argument('--setting_path', type=str, default='./example/config/ad/cmr_generator_setting.json',
                        help='setting dir')
    parser.add_argument('--controllable', type=bool, default=False, help='is controllable')
    FLAGS, _ = parser.parse_known_args()
    return FLAGS

def construct_list_with_score(data_dir, max_time_len):
    user, profile, itm_spar, itm_dens, label, pos, list_len, rank_score = pickle.load(open(data_dir, 'rb'))
    print(len(user), len(itm_spar))
    cut_itm_dens, cut_itm_spar, cut_label, cut_pos, cut_score, cut_usr_spar, cut_usr_dens, de_label, cut_hist_pos = [], [], [], [], [], [], [], [], []
    for i, itm_spar_i, itm_dens_i, label_i, pos_i, list_len_i, score_i in zip(list(range(len(label))),
                                                                     itm_spar, itm_dens, label, pos, list_len, rank_score):

        if len(itm_spar_i) >= max_time_len:
            cut_itm_spar.append(itm_spar_i[: max_time_len])
            cut_itm_dens.append(itm_dens_i[: max_time_len])
            cut_label.append(label_i[: max_time_len])
            # de_label.append(de_lb[: max_time_len])
            cut_pos.append(pos_i[: max_time_len])
            list_len[i] = max_time_len
            cut_score.append(score_i[: max_time_len])
        else:
            cut_itm_spar.append(
                itm_spar_i + [np.zeros_like(np.array(itm_spar_i[0])).tolist()] * (max_time_len - len(itm_spar_i)))
            cut_itm_dens.append(
                itm_dens_i + [np.zeros_like(np.array(itm_dens_i[0])).tolist()] * (max_time_len - len(itm_dens_i)))
            cut_label.append(label_i + [0 for _ in range(max_time_len - list_len_i)])
            cut_score.append(score_i + [float('-inf') for _ in range(max_time_len - list_len_i)])
            # de_label.append(de_lb + [0 for _ in range(max_time_len - list_len_i)])
            cut_pos.append(pos_i + [j for j in range(list_len_i, max_time_len)])

    return user, profile, cut_itm_spar, cut_itm_dens, cut_label, cut_pos, list_len, cut_score

class MMR(object):
    def __init__(self, max_time_len):
        self.max_time_len = max_time_len

    def predict(self, data_batch, lamda=1):
        cate_ids = list(map(lambda a: [i[1] for i in a], data_batch[2]))
        rank_scores = data_batch[-1]
        labels = data_batch[4]
        seq_lens = data_batch[6]
        ret_labels, ret_cates = [], []
        for i in range(len(seq_lens)):
            ret_label, ret_cate = [], []
            cate_set = set()
            cate_id, rank_score, label, seq_len = cate_ids[i], rank_scores[i], labels[i], seq_lens[i]
            mean_score = sum(rank_score[:seq_len])/seq_len
            mask = [0 if i < seq_len else float('-inf') for i in range(self.max_time_len)]
            mmr_score = [rank_score[k] + mask[k] for k in range(self.max_time_len)]
            sorted_idx = sorted(range(self.max_time_len), key=lambda k:mmr_score[k], reverse=True)
            mask[sorted_idx[0]] = float('-inf')
            ret_label.append(label[sorted_idx[0]])
            ret_cate.append(cate_id[sorted_idx[0]])
            cate_set.add(cate_id[sorted_idx[0]])
            for j in range(1, seq_len):
                mmr_score = [mask[k] + lamda * rank_score[k] +
                                        (1 - lamda) * (0 if cate_id[k] in cate_set else abs(mean_score))
                             for k in range(self.max_time_len)]
                sorted_idx = sorted(range(self.max_time_len),
                                    key=lambda k: mmr_score[k],
                                    reverse=True)
                mask[sorted_idx[0]] = float('-inf')
                ret_label.append(label[sorted_idx[0]])
                ret_cate.append(cate_id[sorted_idx[0]])
                cate_set.add(cate_id[sorted_idx[0]])
            ret_labels.append(ret_label)
            ret_cates.append(ret_cate)
        return ret_labels, ret_cates

def agg_func(label1, label2, cate1, cate2):
    pass

def eval_controllable_agg_10(model1, model2, data, batch_size, isrank, metric_scope, _print=False):
    labels = [[] for i in range(11)]
    cates = [[] for i in range(11)]

    data_size = len(data[0])
    batch_num = data_size // batch_size
    print('eval', batch_size, batch_num)

    t = time.time()
    cate_ids = list(map(lambda a: [i[1] for i in a], data[2]))
    for i in range(11):
        for batch_no in range(batch_num):
            data_batch = get_aggregated_batch(data, batch_size=batch_size, batch_no=batch_no)
            label1, cate1 = model1.predict(data_batch, float(i) / 10)
            label2, cate2 = model2.predict(data_batch, float(i) / 10)
            label, cate = agg_func(label1, label2, cate1, cate2)
            labels[i].extend(label)
            # labels.extend(label)
            cates[i].extend(cate)

    res = [[] for i in range(5)]  # [5, 11, 4]
    for label, cate in zip(labels, cates):
        r = evaluate_multi(label, label, cate, metric_scope, isrank, _print)
        for j in range(5):
            res[j].append(r[j])

    print("EVAL TIME: %.4fs" % (time.time() - t))
    return res


if __name__ == '__main__':
    processed_dir = '../Data/toy'
    stat_dir = os.path.join(processed_dir, 'data.stat')
    max_time_len = 10
    initial_ranker = 'lambdaMART'
    reranker='PRM'
    params = reranker_parse_args()

    with open(stat_dir, 'r') as f:
        stat = json.load(f)

    num_item, num_cate, num_ft, profile_fnum, itm_spar_fnum, itm_dens_fnum, = stat['item_num'], stat['cate_num'], \
                                                                              stat['ft_num'], stat['profile_fnum'], \
                                                                              stat['itm_spar_fnum'], stat[
                                                                                  'itm_dens_fnum']
    print('num of item', num_item, 'num of list', stat['train_num'] + stat['val_num'] + stat['test_num'],
          'profile num', profile_fnum, 'spar num', itm_spar_fnum, 'dens num', itm_dens_fnum)

    with open(stat_dir, 'r') as f:
        stat = json.load(f)
    test_dir = os.path.join(processed_dir, initial_ranker + '.data.test')
    test_dir_2 = os.path.join(processed_dir2, initial_ranker + '.data.test')
    if os.path.isfile(test_dir):
        test_lists = pkl.load(open(test_dir, 'rb'))
    else:
        test_lists = construct_list_with_score(os.path.join(processed_dir2, initial_ranker + '.rankings.test'),
                                               max_time_len)
        pkl.dump(test_lists, open(test_dir, 'wb'))
    test_lists2 = pkl.load(open(test_dir_2, 'rb'))
    feature_size, profile_num = num_ft, profile_fnum
    if params.model_type == 'PRM':
        model1 = PRM(feature_size, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum, itm_dens_fnum,
                     profile_num, max_norm=params.max_norm, is_controllable=False, acc_prefer=1)
        model2 = PRM(feature_size, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum, itm_dens_fnum,
                     profile_num, max_norm=params.max_norm, is_controllable=False, acc_prefer=0)
    elif params.model_type == 'miDNN':
        model1 = miDNN(feature_size, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum, itm_dens_fnum,
                       profile_num, max_norm=params.max_norm, acc_prefer=1)
        model2 = miDNN(feature_size, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum, itm_dens_fnum,
                       profile_num, max_norm=params.max_norm, acc_prefer=0)
    elif params.model_type == 'EGR_generator':
        model1 = PPOModel(feature_size, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum, itm_dens_fnum,
                          profile_num, max_norm=params.max_norm, rep_num=params.rep_num, acc_prefer=1,
                          is_controllable=False)
        model2 = PPOModel(feature_size, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum, itm_dens_fnum,
                          profile_num, max_norm=params.max_norm, rep_num=params.rep_num, acc_prefer=0,
                          is_controllable=False)

    elif params.model_type == 'Seq2Slate':
        # model = Seq2Slate(feature_size, eb_dim, hidden_size, max_time_len, max_seq_len, item_fnum, num_cat, mu)
        model1 = SLModel(feature_size, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum, itm_dens_fnum,
                         profile_num, max_norm=params.max_norm, acc_prefer=1, is_controllable=False)
        model2 = SLModel(feature_size, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum, itm_dens_fnum,
                         profile_num, max_norm=params.max_norm, acc_prefer=0, is_controllable=False)
    elif params.model_type == 'CMR':
        model1 = CMR(feature_size, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum, itm_dens_fnum,
                     profile_num, max_norm=params.max_norm, rep_num=params.rep_num, acc_prefer=1, is_controllable=False)
        model2 = CMR(feature_size, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum, itm_dens_fnum,
                     profile_num, max_norm=params.max_norm, rep_num=params.rep_num, acc_prefer=0, is_controllable=False)
    res = eval_controllable_agg_10(model1, model2, test_lists, 16, False, [1, 3, 5, 10], False)
    map_5_l = list(map(lambda a: a[2], res[0]))
    map_l = list(map(lambda a: a[3], res[0]))
    ndcg_5_l = list(map(lambda a: a[2], res[1]))
    ndcg_l = list(map(lambda a: a[3], res[1]))
    ilad_l = list(map(lambda a: a[2], res[3]))
    err_ia_5_l = list(map(lambda a: a[2], res[4]))
    err_ia_l = list(map(lambda a: a[3], res[4]))
    for i in [0, 5, 10]:
        print(map_5_l[i], map_l[i], ndcg_5_l[i], ndcg_l[i], ilad_l[i], err_ia_5_l[i], err_ia_l[i])
    x = [i / 10 for i in range(len(map_l))]
    plt.subplot(2, 2, 1)
    plt.plot(x, map_l, 'r-')
    plt.xlabel('auc_preference')
    plt.ylabel('map')
    plt.subplot(2, 2, 2)
    plt.plot(x, ndcg_l, 'g-')
    plt.xlabel('auc_preference')
    plt.ylabel('ndcg')
    plt.subplot(2, 2, 3)
    plt.plot(x, ilad_l, 'b-')
    plt.xlabel('auc_preference')
    plt.ylabel('ilad')
    plt.subplot(2, 2, 4)
    plt.plot(x, err_ia_l, 'y-')
    plt.xlabel('auc_preference')
    plt.ylabel('err_ia')
    plt.suptitle("{}_{}".format('MMR', 'controllable'))
    plt.legend()
    plt.show()
