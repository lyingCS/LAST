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
from librerank.CMR_generator import *
from librerank.CMR_evaluator import *
from librerank.LAST_generator import *
from librerank.LAST_evaluator import *
import datetime
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def predict(model, data_batch, max_time_len, lamda=1.):
    cate_ids = list(map(lambda a: [i[1] for i in a], data_batch[2]))
    rank_scores, _ = model.eval(data_batch, 9e-5)
    labels = data_batch[4]
    seq_lens = data_batch[6]
    ret_labels, ret_cates, ret_scores, ret_idxs = [], [], [], []
    for i in range(len(seq_lens)):
        ret_label, ret_cate, ret_score, ret_idx = [], [], [], []
        cate_set = set()
        cate_id, rank_score, label, seq_len = cate_ids[i], rank_scores[i], labels[i], seq_lens[i]
        mean_score = sum(rank_score[:seq_len]) / seq_len
        pre_score = sorted(rank_score[:seq_len])
        diff_score = (pre_score[-1] - pre_score[0])/2
        mask = [0 if i < seq_len else float('-inf') for i in range(max_time_len)]
        mmr_score = [rank_score[k] + mask[k] for k in range(max_time_len)]
        sorted_idx = sorted(range(max_time_len), key=lambda k: mmr_score[k], reverse=True)
        mask[sorted_idx[0]] = float('-inf')
        ret_label.append(label[sorted_idx[0]])
        ret_cate.append(cate_id[sorted_idx[0]])
        ret_score.append(mmr_score[sorted_idx[0]])
        ret_idx.append(sorted_idx[0])
        cate_set.add(cate_id[sorted_idx[0]])
        for j in range(1, seq_len):
            mmr_score = [mask[k] + lamda * rank_score[k] +
                         (1 - lamda) * (0 if cate_id[k] in cate_set else abs(diff_score))
                         for k in range(max_time_len)]
            sorted_idx = sorted(range(max_time_len),
                                key=lambda k: mmr_score[k],
                                reverse=True)
            mask[sorted_idx[0]] = float('-inf')
            ret_label.append(label[sorted_idx[0]])
            ret_cate.append(cate_id[sorted_idx[0]])
            ret_score.append(mmr_score[sorted_idx[0]])
            ret_idx.append(sorted_idx[0])
            cate_set.add(cate_id[sorted_idx[0]])
        ret_idx += [0 for i in range(max_time_len - len(ret_idx))]
        ret_labels.append(ret_label)
        ret_cates.append(ret_cate)
        ret_scores.append(ret_score)
        ret_idxs.append(ret_idx)
    return ret_labels, ret_cates, ret_scores, ret_idxs


def eval_controllable_10(params, model, data, batch_size, isrank, metric_scope, evaluator, _print=False):
    n = 11
    labels = [[] for i in range(n)]
    cates = [[] for i in range(n)]
    preds = [[] for i in range(n)]
    idxs = [[] for i in range(n)]

    data_size = len(data[0])
    batch_num = data_size // batch_size
    print('eval', batch_size, batch_num)

    t = time.time()
    cate_ids = list(map(lambda a: [i[1] for i in a], data[2]))
    for i in range(n):
        for batch_no in range(batch_num):
            data_batch = get_aggregated_batch(data, batch_size=batch_size, batch_no=batch_no)
            label, cate, pred, idx = predict(model, data_batch, params.max_time_len, float(i)/10)
            labels[i].extend(label)
            # labels.extend(label)
            cates[i].extend(cate)
            preds[i].extend(pred)
            idxs[i].extend(idx)

    res = [[] for i in range(6)]  # [5, 11, 4]
    pidx = 0
    for label, cate, pred, idx in zip(labels, cates, preds, idxs):
        r = evaluate_multi(label, pred, cate, metric_scope, isrank, _print)
        for j in range(5):
            res[j].append(r[j])
        eva_list = [[] for i in range(len(metric_scope))]
        for batch_no in range(batch_num):
            data_batch = get_aggregated_batch(data, batch_size=batch_size, batch_no=batch_no)
            order_batch = get_aggregated_batch([idx], batch_size=batch_size, batch_no=batch_no)[0]
            batch_sum, batch_ave = evaluator_metrics(data_batch, order_batch, metric_scope, model,
                                                     evaluator)  # [scope_num, B]
            for i in range(len(metric_scope)):
                eva_list[i].extend(batch_ave[i])
        # res[5].append(np.mean(np.array(eva_list), axis=1))
        print(pidx)
        for i, s in enumerate(metric_scope):
            # eva = np.mean(np.array(eva_list[i]))
            print("@%d  MAP: %.4f  NDCG: %.4f  CLICKS: %.4f  ERR_IA: %.4f  EVA_AVE: %.4f" % (
                s, res[0][pidx][i], res[1][pidx][i], res[2][pidx][i], res[4][pidx][i], np.mean(np.array(eva_list[i]))))
        pidx += 1

    print("EVAL TIME: %.4fs" % (time.time() - t))
    return res


def reranker_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_time_len', default=10, type=int, help='max time length')
    parser.add_argument('--save_dir', type=str, default='./', help='dir that saves logs and model')
    parser.add_argument('--data_dir', type=str, default='./data/ad/', help='data dir')
    parser.add_argument('--model_type', default='Seq2Slate',
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
    parser.add_argument('--evaluator_metrics_path', type=str,
                        default="./model//save_model_ad/10/202303091111_lambdaMART_LAST_evaluator_16_0.0005_0.0002_64_16_0.8_1.0",
                        help='evaluator model')
    parser.add_argument('--reload_path', type=str,
                        default='./model/save_model_ad/10/202303211923_lambdaMART_SetRank_32_5e-05_0.0001_64_16_0.8_1.0',
                        help='model ckpt dir')
    # 202303101454_lambdaMART_miDNN_64_0.0001_9e-05_64_16_0.8_1.0
    # 202303131350_lambdaMART_GSF_32_5e-05_0.0001_64_16_0.8_1.0
    # 202303131355_lambdaMART_DLCM_32_0.0001_0.0001_64_16_0.8_1.0
    # 202303211923_lambdaMART_SetRank_32_5e-05_0.0001_64_16_0.8_1.0
    # 202303161405_lambdaMART_PRM_32_5e-05_0.0001_64_16_0.8_1.0
    parser.add_argument('--setting_path', type=str, default='./example/config/ad/setrank_setting.json',
                        help='setting dir')
    parser.add_argument('--controllable', type=bool, default=False, help='is controllable')
    FLAGS, _ = parser.parse_known_args()
    return FLAGS


if __name__ == '__main__':
    parse = reranker_parse_args()
    if parse.setting_path:
        parse = load_parse_from_json(parse, parse.setting_path)
    params = parse
    # processed_dir = '../Data/toy'
    processed_dir = 'Data/ad'
    stat_dir = os.path.join(processed_dir, 'data.stat')
    max_time_len = 10
    initial_ranker = 'lambdaMART'
    with open(stat_dir, 'r') as f:
        stat = json.load(f)

    num_item, num_cate, feature_size, profile_num, itm_spar_fnum, itm_dens_fnum, = stat['item_num'], stat['cate_num'], \
                                                                                   stat['ft_num'], stat['profile_fnum'], \
                                                                                   stat['itm_spar_fnum'], stat[
                                                                                       'itm_dens_fnum']
    print('num of item', num_item, 'num of list', stat['train_num'] + stat['val_num'] + stat['test_num'],
          'profile num', profile_num, 'spar num', itm_spar_fnum, 'dens num', itm_dens_fnum)

    with open(stat_dir, 'r') as f:
        stat = json.load(f)
    test_dir = os.path.join(processed_dir, initial_ranker + '.data.test')
    # test_dir_2 = os.path.join(processed_dir2, initial_ranker + '.data.test')
    test_lists = pkl.load(open(test_dir, 'rb'))
    # test_lists2 = pkl.load(open(test_dir_2, 'rb'))

    tf.reset_default_graph()

    # gpu settings
    gpu_options = tf.GPUOptions(allow_growth=True)

    perlist = False
    if params.model_type == 'PRM':
        model = PRM(feature_size, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum, itm_dens_fnum,
                    profile_num, max_norm=params.max_norm, is_controllable=params.controllable,
                    acc_prefer=params.acc_prefer)
    elif params.model_type == 'SetRank':
        model = SetRank(feature_size, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum, itm_dens_fnum,
                        profile_num, max_norm=params.max_norm)
    elif params.model_type == 'DLCM':
        model = DLCM(feature_size, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum, itm_dens_fnum,
                     profile_num, max_norm=params.max_norm, acc_prefer=params.acc_prefer)
    elif params.model_type == 'GSF':
        model = GSF(feature_size, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum, itm_dens_fnum,
                    profile_num, max_norm=params.max_norm, group_size=params.group_size)
    elif params.model_type == 'miDNN':
        model = miDNN(feature_size, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum, itm_dens_fnum,
                      profile_num, max_norm=params.max_norm, is_controllable=params.controllable,
                      acc_prefer=params.acc_prefer)
    elif params.model_type == 'EGR_evaluator':
        model = EGR_evaluator(feature_size, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum,
                              itm_dens_fnum,
                              profile_num, max_norm=params.max_norm)
    elif params.model_type == 'CMR_evaluator':
        model = CMR_evaluator(feature_size, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum,
                              itm_dens_fnum,
                              profile_num, max_norm=params.max_norm)
    elif params.model_type == 'EGR_generator':
        model = PPOModel(feature_size, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum, itm_dens_fnum,
                         profile_num, max_norm=params.max_norm, rep_num=params.rep_num, acc_prefer=params.acc_prefer,
                         is_controllable=params.controllable)
    elif params.model_type == 'Seq2Slate':
        # model = Seq2Slate(feature_size, eb_dim, hidden_size, max_time_len, max_seq_len, item_fnum, num_cat, mu)
        model = SLModel(feature_size, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum, itm_dens_fnum,
                        profile_num, max_norm=params.max_norm, acc_prefer=params.acc_prefer,
                        is_controllable=params.controllable)
    elif params.model_type == 'CMR_generator':
        model = CMR_generator(feature_size, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum,
                              itm_dens_fnum,
                              profile_num, max_norm=params.max_norm, rep_num=params.rep_num,
                              acc_prefer=params.acc_prefer,
                              is_controllable=params.controllable)
    else:
        print('No Such Model', params.model_type)
        exit(0)

    with model.graph.as_default() as g:
        sess = tf.Session(graph=g, config=tf.ConfigProto(gpu_options=gpu_options))
        model.set_sess(sess)
        sess.run(tf.global_variables_initializer())
        model.load(params.reload_path)

    evaluator = LAST_evaluator(feature_size, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum,
                               itm_dens_fnum,
                               profile_num, max_norm=params.max_norm)
    with evaluator.graph.as_default() as g:
        sess = tf.Session(graph=g, config=tf.ConfigProto(gpu_options=gpu_options))
        evaluator.set_sess(sess)
        sess.run(tf.global_variables_initializer())
        evaluator.load(params.evaluator_metrics_path)

    res = eval_controllable_10(params, model, test_lists, 16, False, [1, 3, 5, 10], evaluator, False)
