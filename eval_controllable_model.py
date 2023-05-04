import os
# from sklearn.metrics import log_loss, roc_auc_score
import random
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from librerank.utils import *
from librerank.reranker import *
from librerank.CMR_generator import *
from librerank.CMR_evaluator import *
from librerank.rl_reranker import *
import datetime
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import json


# import seaborn as sns

def eval_controllable_10(params, model, data, l2_reg, batch_size, isrank, metric_scope, _print=False):
    preds = [[] for i in range(11)]
    # labels = []
    losses = [[] for i in range(11)]
    auc_losses = [[] for i in range(11)]
    div_losses = [[] for i in range(11)]

    data_size = len(data[0])
    batch_num = data_size // batch_size
    print('eval', batch_size, batch_num)

    t = time.time()
    # cates = np.reshape(np.array(data[1])[:, :, 1], [-1, max_time_len]).tolist()
    labels = data[4]
    # print(preds[0], labels[0])
    # poss = data[-2]
    cate_ids = list(map(lambda a: [i[1] for i in a], data[2]))
    for i in range(11):
        print(i)
        for batch_no in range(batch_num):
            data_batch = get_aggregated_batch(data, batch_size=batch_size, batch_no=batch_no)
            pred, loss = model.eval(data_batch, l2_reg, float(i) / 10)
            preds[i].extend(pred)
            # labels.extend(label)
            losses[i].append(loss)
            train_prefer = float(i)/10
            # if params.model_type == 'EGR_generator':
            #
            #     data_batch = repeat_data(data_batch, params.rep_num)
            #
            #     act_idx_out, act_probs_one, rl_sp_outputs, rl_de_outputs, mask_arr, lp_sp_data, lp_de_data, _, enc_input, \
            #     cate_chosen, cate_seq = model.predict(data_batch, train_prefer, params.l2_reg)
            #
            #     auc_rewards = evaluator.predict(rl_sp_outputs, rl_de_outputs, data_batch[6])
            #
            #     _, div_rewards = model.build_erria_reward(cate_chosen, cate_seq)
            #
            #     auc_loss, div_loss = model.count_loss(data_batch, rl_sp_outputs, rl_de_outputs,
            #                                                         act_probs_one, act_idx_out,
            #                                                         auc_rewards, div_rewards, mask_arr,
            #                                                         params.c_entropy, params.lr,
            #                                                         params.l2_reg,
            #                                                         params.keep_prob, train_prefer=train_prefer)
            #     auc_losses[i].append(auc_loss)
            #     div_losses[i].append(div_loss)
            # elif params.model_type == 'CMR_generator':
            #     training_attention_distribution, training_prediction_order, predictions, cate_seq, cate_chosen = \
            #         model.rerank(data_batch, params.keep_prob, train_prefer=float(i)/10)
            #     rl_sp_outputs, rl_de_outputs = model.build_ft_chosen(data_batch, training_prediction_order)
            #     rerank_click = np.array(model.build_label_reward(data_batch[4], training_prediction_order))
            #     auc_rewards = evaluator.predict(np.array(data_batch[1]), rl_sp_outputs, rl_de_outputs,
            #                                         data_batch[6])
            #     base_auc_rewards = evaluator.predict(np.array(data_batch[1]), np.array(data_batch[2]),
            #                                              np.array(data_batch[3]), data_batch[6])
            #     auc_rewards -= base_auc_rewards
            #
            #     _, base_div_rewards = model.build_erria_reward(cate_seq, cate_seq)  # rank base rerank new
            #     _, div_rewards = model.build_erria_reward(cate_chosen, cate_seq)
            #     div_rewards -= base_div_rewards
            #
            #     auc_loss, div_loss = model.count_loss(data_batch, training_prediction_order, auc_rewards, div_rewards,
            #                                            params.lr, params.l2_reg, params.keep_prob,
            #                                            train_prefer=float(i)/10)
            #     auc_losses[i].append(auc_loss)
            #     div_losses[i].append(div_loss)
            # elif params.model_type == 'Seq2Slate':  # [B, N, N]    [[0,0,1],[0,1,0],[1,0,0]]
            #     act_idx_out, act_probs_one, rl_sp_outputs, rl_de_outputs, mask_arr, lp_sp_data, lp_de_data, _, enc_input, \
            #     cate_chosen, cate_seq \
            #         = model.predict(data_batch, float(i)/10, params.l2_reg)
            #     mask_arr[mask_arr < 0] = 0
            #     div_label, _ = model.build_erria_reward(cate_chosen, cate_seq)
            #     auc_loss, div_loss = model.count_loss(data_batch, rl_sp_outputs, rl_de_outputs, mask_arr, params.lr,
            #                        params.l2_reg, div_label, params.keep_prob, train_prefer=float(i)/10)
            #     auc_losses[i].append(auc_loss)
            #     div_losses[i].append(div_loss)

    # print(np.mean(auc_losses, axis=-1))
    # print(np.mean(div_losses, axis=-1))

    loss = [sum(loss) / len(loss) for loss in losses]  # [11]

    res = [[] for i in range(5)]  # [5, 11, 4]
    auc_l = []
    idx = 0
    for pred in preds:
        print(idx)
        r = evaluate_multi(labels, pred, cate_ids, metric_scope, isrank, _print)
        auc, _ = evaluate_auc(pred, labels)
        for j in range(5):
            res[j].append(r[j])
        auc_l.append(auc)
        idx += 1

    print("EVAL TIME: %.4fs" % (time.time() - t))
    # return loss, res_low, res_high
    return loss, res, auc_l, auc_losses, div_losses


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
    #1234
    # parser.add_argument('--reload_path', type=str, default='./model/save_model_ad/10/202301181615_lambdaMART_CMR_generator_16_0.0001_0.0001_64_16_0.8_controllable', help='model ckpt dir') #38081
    # parser.add_argument('--reload_path', type=str, default='./model/save_model_ad/10/202304092152_lambdaMART_CMR_generator_16_0.0001_0.0001_64_16_0.8_controllable', help='model ckpt dir') #38081
    # parser.add_argument('--reload_path', type=str, default='./model/save_model_ad/10/202301122214_lambdaMART_Seq2Slate_16_0.0001_0.0001_64_16_0.8_controllable/', help='model ckpt dir')
    # parser.add_argument('--reload_path', type=str, default='./model/save_model_ad/10/202301162239_lambdaMART_EGR_generator_16_0.0001_0.0001_64_16_0.8_controllable', help='model ckpt dir')
    #2234
    # parser.add_argument('--reload_path', type=str, default='./model//save_model_ad/10/202304072246_lambdaMART_CMR_generator_16_0.0001_0.0001_64_16_0.8_controllable', help='model ckpt dir')
    # parser.add_argument('--reload_path', type=str, default='./model/save_model_ad/10/202304071858_lambdaMART_Seq2Slate_16_0.0001_0.0001_64_16_0.8_controllable', help='model ckpt dir')
    # parser.add_argument('--reload_path', type=str, default='./model/save_model_ad/10/202304071856_lambdaMART_EGR_generator_16_0.0001_0.0001_64_16_0.8_controllable', help='model ckpt dir')
    #3234
    # parser.add_argument('--reload_path', type=str, default='./model//save_model_ad/10/202304081644_lambdaMART_CMR_generator_16_0.0001_0.0001_64_16_0.8_controllable', help='model ckpt dir')
    # parser.add_argument('--reload_path', type=str, default='./model/save_model_ad/10/202304072055_lambdaMART_Seq2Slate_16_0.0001_0.0001_64_16_0.8_controllable', help='model ckpt dir')
    # parser.add_argument('--reload_path', type=str, default='./model/save_model_ad/10/202304072319_lambdaMART_EGR_generator_16_0.0001_0.0001_64_16_0.8_controllable', help='model ckpt dir')

    # parser.add_argument('--reload_path', type=str, default='./model/save_model_ad/10/202304101710_lambdaMART_EGR_generator_16_0.0001_0.0001_64_16_0.8_controllable', help='model ckpt dir')
    # parser.add_argument('--reload_path', type=str, default='./model/save_model_ad/10/202304101659_lambdaMART_EGR_generator_16_0.0001_0.0001_64_16_0.8_controllable', help='model ckpt dir')
    # parser.add_argument('--reload_path', type=str, default='./model/save_model_ad/10/202304101946_lambdaMART_CMR_generator_16_0.0001_0.0001_64_16_0.8_controllable', help='model ckpt dir') #38081
    parser.add_argument('--reload_path', type=str, default='./model/save_model_ad/10/202304130851_lambdaMART_CMR_generator_16_0.0001_0.0001_64_16_0.8_controllable', help='model ckpt dir') #38081

    # parser.add_argument('--setting_path', type=str, default='./config/prm_setting.json', help='setting dir')
    parser.add_argument('--setting_path', type=str, default='./example/config/ad/cmr_generator_setting.json',
                help='setting dir')
    parser.add_argument('--controllable', type=bool, default=False, help='is controllable')
    FLAGS, _ = parser.parse_known_args()
    return FLAGS


if __name__ == '__main__':
    parse = reranker_parse_args()
    if parse.setting_path:
        parse = load_parse_from_json(parse, parse.setting_path)
    data_set_name = parse.data_set_name
    processed_dir = parse.data_dir
    stat_dir = os.path.join(processed_dir, 'data.stat')
    max_time_len = parse.max_time_len
    initial_ranker = parse.initial_ranker
    if data_set_name == 'prm' and parse.max_time_len > 30:
        max_time_len = 30
    print(parse)
    with open(stat_dir, 'r') as f:
        stat = json.load(f)

    num_item, num_cate, num_ft, profile_fnum, itm_spar_fnum, itm_dens_fnum, = stat['item_num'], stat['cate_num'], \
                                                                              stat['ft_num'], stat['profile_fnum'], \
                                                                              stat['itm_spar_fnum'], stat[
                                                                                  'itm_dens_fnum']
    print('num of item', num_item, 'num of list', stat['train_num'] + stat['val_num'] + stat['test_num'],
          'profile num', profile_fnum, 'spar num', itm_spar_fnum, 'dens num', itm_dens_fnum)

    params = parse
    with open(stat_dir, 'r') as f:
        stat = json.load(f)
    test_dir = os.path.join(processed_dir, initial_ranker + '.data.test')
    if os.path.isfile(test_dir):
        test_lists = pkl.load(open(test_dir, 'rb'))
    else:
        test_lists = construct_list(os.path.join(processed_dir, initial_ranker + '.rankings.test'), max_time_len)
        pkl.dump(test_lists, open(test_dir, 'wb'))

    gpu_options = tf.GPUOptions(allow_growth=True)
    if params.model_type == 'CMR_generator':
        model = CMR_generator(num_ft, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum, itm_dens_fnum,
                          profile_fnum, max_norm=params.max_norm, rep_num=params.rep_num, acc_prefer=params.acc_prefer,
                      is_controllable=params.controllable)
        if params.evaluator_type == 'cmr':
            evaluator = CMR_evaluator(num_ft, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum,
                                      itm_dens_fnum,
                                      profile_fnum, max_norm=params.max_norm)
        with evaluator.graph.as_default() as g:
            sess = tf.Session(graph=g, config=tf.ConfigProto(gpu_options=gpu_options))
            evaluator.set_sess(sess)
            sess.run(tf.global_variables_initializer())
            evaluator.load(params.evaluator_path)
    elif params.model_type == 'Seq2Slate':
        model = SLModel(num_ft, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum, itm_dens_fnum,
                        profile_fnum, max_norm=params.max_norm, acc_prefer=params.acc_prefer,
                        is_controllable=params.controllable)
    elif params.model_type == 'EGR_generator':
        model = PPOModel(num_ft, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum, itm_dens_fnum,
                        profile_fnum, max_norm=params.max_norm, rep_num=params.rep_num, acc_prefer=params.acc_prefer,
                        is_controllable=params.controllable)
        evaluator = EGR_evaluator(num_ft, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum,
                                  itm_dens_fnum,
                                  profile_fnum, max_norm=params.max_norm)
        with evaluator.graph.as_default() as g:
            sess = tf.Session(graph=g, config=tf.ConfigProto(gpu_options=gpu_options))
            evaluator.set_sess(sess)
            sess.run(tf.global_variables_initializer())
            evaluator.load(params.evaluator_path)
    elif params.model_type == 'PRM':
        model = PRM(num_ft, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum, itm_dens_fnum,
                    profile_fnum, max_norm=params.max_norm, is_controllable=params.controllable,
                    acc_prefer=params.acc_prefer)
    elif params.model_type == 'miDNN':
        model = miDNN(num_ft, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum, itm_dens_fnum,
                    profile_fnum, max_norm=params.max_norm, is_controllable=params.controllable,
                    acc_prefer=params.acc_prefer)
    with model.graph.as_default() as g:
        sess = tf.Session(graph=g, config=tf.ConfigProto(gpu_options=gpu_options))
        model.set_sess(sess)
        sess.run(tf.global_variables_initializer())
        model.load(params.reload_path)

    loss, res, auc, auc_losses, div_losses = eval_controllable_10(params, model, test_lists, params.l2_reg, params.batch_size, True,
                                     params.metric_scope, False)
    map_5_l = list(map(lambda a: a[2], res[0]))
    map_l = list(map(lambda a: a[3], res[0]))
    ndcg_5_l = list(map(lambda a: a[2], res[1]))
    ndcg_l = list(map(lambda a: a[3], res[1]))
    ilad_l = list(map(lambda a: a[2], res[3]))
    err_ia_5_l = list(map(lambda a: a[2], res[4]))
    err_ia_l = list(map(lambda a: a[3], res[4]))
    all_data_dict = {"all_data": [map_5_l, map_l, ndcg_5_l, ndcg_l, ilad_l, err_ia_5_l, err_ia_l, auc]}
    print(all_data_dict["all_data"])
    for i in [0, 5, 10]:
        print("%.5f" % map_5_l[i], "%.5f" % map_l[i], "%.5f" % ndcg_5_l[i], "%.5f" % ndcg_l[i], "%.5f" % ilad_l[i],
              "%.5f" % err_ia_5_l[i], "%.5f" % err_ia_l[i], "%.5f" % auc[i])
    x = [i / 10 for i in range(len(map_l))]
    # print(auc_losses)
    # print(div_losses)
    # sns.set(style="darkgrid")
    # fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    #
    # # 绘制第一幅图
    # sns.lineplot(x=x, y=map_5_l, ax=axes[0, 0])
    # axes[0, 0].set(xlabel='auc preference', ylabel='MAP@5')
    #
    # # 绘制第二幅图
    # sns.lineplot(x=x, y=ndcg_5_l, ax=axes[0, 1])
    # axes[0, 1].set(xlabel='auc preference', ylabel='NDCG@5')
    #
    # # 绘制第三幅图
    # sns.lineplot(x=x, y=ilad_l, ax=axes[1, 0])
    # axes[1, 0].set(xlabel='auc preference', ylabel='ILAD@5')
    #
    # # 绘制第四幅图
    # sns.lineplot(x=x, y=err_ia_5_l, ax=axes[1, 1])
    # axes[1, 1].set(xlabel='auc preference', ylabel='ERR_IA@5')
    #
    # # 调整子图间距
    # plt.tight_layout()
    #
    # # 添加标题
    # plt.suptitle("{}_{}".format('CMR_generator', 'controllable'))
    #
    # # 显示图形
    # plt.show()

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
    plt.suptitle("{}_{}".format('CMR_generator', 'controllable'))
    plt.legend()
    plt.tight_layout()
    plt.show()
    # with open(params.reload_path + '/controllable_data.json', 'wb') as file:
    #     pickle.dump(all_data_dict, file)
