import os
# from sklearn.metrics import log_loss, roc_auc_score
import random
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from librerank.utils import *
from librerank.reranker import *
from librerank.rl_reranker import *
from librerank.CMR_generator import *
from librerank.CMR_evaluator import *
from librerank.LAST_generator import *
from librerank.LAST_evaluator import *
import datetime


def eval(model, data, l2_reg, batch_size, isrank, metric_scope, _print=False,
         with_evaluator=False, evaluator=None):
    preds = []
    # labels = []
    losses = []

    data_size = len(data[0])
    batch_num = data_size // batch_size
    print('eval', batch_size, batch_num)

    if with_evaluator:
        evaluator_sum, evaluator_ave = [[] for _ in range(len(metric_scope))], [[] for _ in range(len(metric_scope))]
    t = time.time()
    for batch_no in range(batch_num):
        data_batch = get_aggregated_batch(data, batch_size=batch_size, batch_no=batch_no)
        pred, loss = model.eval(data_batch, l2_reg)
        preds.extend(pred)
        # labels.extend(label)
        losses.append(loss)
        if with_evaluator:
            order = [sorted(range(len(_pred)), key=lambda k: _pred[k], reverse=True) for _pred in pred]
            batch_sum, batch_ave = evaluator_metrics(data_batch, order, metric_scope, model, evaluator)
            for i in range(len(metric_scope)):
                evaluator_sum[i].extend(batch_sum[i])
                evaluator_ave[i].extend(batch_ave[i])

    loss = sum(losses) / len(losses)
    # cates = np.reshape(np.array(data[1])[:, :, 1], [-1, max_time_len]).tolist()
    labels = data[4]
    # print(preds[0], labels[0])
    # poss = data[-2]
    cate_ids = list(map(lambda a: [i[1] for i in a], data[2]))

    res = list(evaluate_multi(labels, preds, cate_ids, metric_scope, isrank, _print))
    if with_evaluator:
        res.append(np.mean(np.array(evaluator_sum), axis=-1))
        res.append(np.mean(np.array(evaluator_ave), axis=-1))

    print("EVAL TIME: %.4fs" % (time.time() - t))
    # return loss, res_low, res_high
    return loss, res


def eval_pv_evaluator(model, data, l2_reg, batch_size, isrank, metric_scope, _print=False):
    preds = []
    # labels = []
    losses = []
    res = [[] for i in range(5)]  # [5, 4]

    data_size = len(data[0])
    batch_num = data_size // batch_size
    clicks = []
    print('eval', batch_size, batch_num)

    t = time.time()
    for batch_no in range(batch_num):
        data_batch = get_aggregated_batch(data, batch_size=batch_size, batch_no=batch_no)
        pred, loss, b_s = model.eval(data_batch, l2_reg)
        preds.append(pred)
        # labels.extend(label)
        losses.append(loss)
        clicks.append(pv_process_reward(data_batch))

    loss = sum(losses) / len(losses)
    for i in range(5):
        res[i] = np.array([-loss for i in range(4)])

    print("EVAL TIME: %.4fs" % (time.time() - t))
    return loss, res


def eval_controllable(model, data, l2_reg, batch_size, isrank, metric_scope, _print=False):
    preds = [[] for i in range(3)]
    # labels = []
    losses = [[] for i in range(3)]

    data_size = len(data[0])
    batch_num = data_size // batch_size
    print('eval', batch_size, batch_num)

    t = time.time()
    # cates = np.reshape(np.array(data[1])[:, :, 1], [-1, max_time_len]).tolist()
    labels = data[4]
    # print(preds[0], labels[0])
    # poss = data[-2]
    cate_ids = list(map(lambda a: [i[1] for i in a], data[2]))
    # for i in range(11):
    for i in range(3):
        for batch_no in range(batch_num):
            data_batch = get_aggregated_batch(data, batch_size=batch_size, batch_no=batch_no)
            # pred, loss = model.eval(data_batch, l2_reg, float(i)/10)
            pred, loss = model.eval(data_batch, l2_reg, float(i * 5) / 10)
            preds[i].extend(pred)
            # labels.extend(label)
            losses[i].append(loss)

    loss = [sum(loss) / len(loss) for loss in losses]  # [11]

    res = [[] for i in range(5)]  # [5, 11, 4]
    for pred in preds:
        r = evaluate_multi(labels, pred, cate_ids, metric_scope, isrank, _print)
        for j in range(5):
            res[j].append(r[j])

    print("EVAL TIME: %.4fs" % (time.time() - t))
    # return loss, res_low, res_high
    return loss, res


def train(train_file, test_file, feature_size, max_time_len, itm_spar_fnum, itm_dens_fnum, profile_num, params):
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
    elif params.model_type == 'LAST_evaluator':
        model = LAST_evaluator(feature_size, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum,
                               itm_dens_fnum,
                               profile_num, max_norm=params.max_norm)
    elif params.model_type == 'EGR_generator':
        model = PPOModel(feature_size, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum, itm_dens_fnum,
                         profile_num, max_norm=params.max_norm, rep_num=params.rep_num, acc_prefer=params.acc_prefer,
                         is_controllable=params.controllable)
        # discriminator = EGR_discriminator(feature_size, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum, itm_dens_fnum,
        #              profile_num, max_norm=params.max_norm)
        evaluator = EGR_evaluator(feature_size, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum,
                                  itm_dens_fnum,
                                  profile_num, max_norm=params.max_norm)
        with evaluator.graph.as_default() as g:
            sess = tf.Session(graph=g, config=tf.ConfigProto(gpu_options=gpu_options))
            evaluator.set_sess(sess)
            sess.run(tf.global_variables_initializer())
            evaluator.load(params.evaluator_path)
        # [1,1,1,0,0,0]
        # [1,2,3,0,0,0,0,0,0]
        # [1,0,0,2,0,0,0,3,0]
        # [0,1]
        # [0.1 0.1 0.1 0 0 0 0 0 0 0]
        # with discriminator.graph.as_default() as g:
        #     sess = tf.Session(graph=g, config=tf.ConfigProto(gpu_options=gpu_options))
        #     discriminator.set_sess(sess)
        #     sess.run(tf.global_variables_initializer())
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
        if params.evaluator_type == 'cmr':
            evaluator = CMR_evaluator(feature_size, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum,
                                      itm_dens_fnum,
                                      profile_num, max_norm=params.max_norm)
        elif params.evaluator_type == 'egr':
            evaluator = EGR_evaluator(feature_size, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum,
                                      itm_dens_fnum,
                                      profile_num, max_norm=params.max_norm)
        elif params.evaluator_type == 'last':
            evaluator = LAST_evaluator(feature_size, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum,
                                       itm_dens_fnum,
                                       profile_num, max_norm=params.max_norm)
        with evaluator.graph.as_default() as g:
            sess = tf.Session(graph=g, config=tf.ConfigProto(gpu_options=gpu_options))
            evaluator.set_sess(sess)
            sess.run(tf.global_variables_initializer())
            evaluator.load(params.evaluator_path)
    elif params.model_type == 'LAST_generator':
        model = LAST_generator(feature_size, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum,
                               itm_dens_fnum,
                               profile_num, max_norm=params.max_norm, rep_num=params.rep_num,
                               acc_prefer=params.acc_prefer,
                               is_controllable=False)
        evaluator = LAST_evaluator(feature_size, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum,
                                   itm_dens_fnum,
                                   profile_num, max_norm=params.max_norm)
        with evaluator.graph.as_default() as g:
            sess = tf.Session(graph=g, config=tf.ConfigProto(gpu_options=gpu_options))
            evaluator.set_sess(sess)
            sess.run(tf.global_variables_initializer())
            evaluator.load(params.evaluator_path)
    else:
        print('No Such Model', params.model_type)
        exit(0)

    with model.graph.as_default() as g:
        sess = tf.Session(graph=g, config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        model.set_sess(sess)
        if params.reload_model == True:
            model.load(params.reload_path)
            print("reload ", params.reload_path)

    # training_monitor = {
    #     'train_loss': [],
    #     'vali_loss': [],
    #     'map_l': [],
    #     'ndcg_l': [],
    #     'clicks_l': [],
    #     'utility_l': [],
    #     'map_h':[],
    #     'ndcg_h': [],
    #     'clicks_h': [],
    #     'utility_h': [],
    # }
    training_monitor = {
        'train_loss': [],
        'auc_train_loss': [],
        'div_train_loss': [],
        'train_prefer': [],
        'vali_loss': [],
        'map_l': [],
        'ndcg_l': [],
        'clicks_l': [],
        'ilad_l': [],
        'err_ia_l': [],
        'alpha_ndcg': [],
        'eva_sum': [],
        'eva_ave': [],
        # 'utility_l': [],
        # 'de_ndcg_l': [],
    }

    training_monitor_2 = {
        'train_loss': [],
        'auc_train_loss': [],
        'div_train_loss': [],
        'train_prefer': [],
        'vali_loss': [],
        'map_l': [],
        'ndcg_l': [],
        'clicks_l': [],
        'ilad_l': [],
        'err_ia_l': [],
        'alpha_ndcg': [],
        'eva_sum': [],
        'eva_ave': [],
    }

    model_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(params.timestamp, initial_ranker, params.model_type,
                                                        params.batch_size,
                                                        params.lr, params.l2_reg, params.hidden_size, params.eb_dim,
                                                        params.keep_prob,
                                                        "controllable" if params.controllable else params.acc_prefer)
    if not os.path.exists('{}/logs_{}/{}'.format(parse.save_dir, data_set_name, max_time_len)):
        os.makedirs('{}/logs_{}/{}'.format(parse.save_dir, data_set_name, max_time_len))
    if not os.path.exists('{}/save_model_{}/{}/{}/'.format(parse.save_dir, data_set_name, max_time_len, model_name)):
        os.makedirs('{}/save_model_{}/{}/{}/'.format(parse.save_dir, data_set_name, max_time_len, model_name))
    save_path = '{}/save_model_{}/{}/{}/ckpt'.format(parse.save_dir, data_set_name, max_time_len, model_name)
    log_save_path = '{}/logs_{}/{}/{}.metrics'.format(parse.save_dir, data_set_name, max_time_len, model_name)

    # training process
    # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    #     sess.run(tf.global_variables_initializer())
    #     sess.run(tf.local_variables_initializer())

    train_losses_step = []
    auc_train_losses_step = []
    div_train_losses_step = []
    train_prefer_step = []

    # before training process
    step = 0

    if True:
        if not params.controllable:
            # pass
            # if params.model_type == 'CMR_evaluator':
            #     vali_loss, res = eval_pv_evaluator(model, test_file, params.l2_reg, params.batch_size, False,
            #                                        params.metric_scope)
            # else:
            # if params.model_type == 'LAST_generator':
            #     vali_loss, res = eval_last(model, test_file, params.l2_reg, params.batch_size, False,
            #                    params.metric_scope, with_evaluator=params.with_evaluator_metrics, evaluator=evaluator)
            # else:
            vali_loss, res = eval(model, test_file, params.l2_reg, params.batch_size, False,
                                  params.metric_scope, with_evaluator=params.with_evaluator_metrics, evaluator=
                                  evaluator if params.with_evaluator_metrics else None)
            training_monitor['train_loss'].append(None)
            training_monitor['vali_loss'].append(None)
            training_monitor['map_l'].append(res[0][0])
            training_monitor['ndcg_l'].append(res[1][0])
            # training_monitor['de_ndcg_l'].append(res[2][0])
            training_monitor['clicks_l'].append(res[2][0])
            training_monitor['ilad_l'].append(res[3][0])
            training_monitor['err_ia_l'].append(res[4][0])
            training_monitor['alpha_ndcg'].append(res[5][0])
            # training_monitor['utility_l'].append(res[4][0])
            if params.with_evaluator_metrics:
                training_monitor['eva_sum'].append(res[-2][0])
                training_monitor['eva_ave'].append(res[-1][0])

            training_monitor_2['train_loss'].append(None)
            training_monitor_2['vali_loss'].append(None)
            training_monitor_2['map_l'].append(res[0].tolist())
            training_monitor_2['ndcg_l'].append(res[1].tolist())
            # training_monitor['de_ndcg_l'].append(res[2][0])
            training_monitor_2['clicks_l'].append(res[2].tolist())
            training_monitor_2['ilad_l'].append(res[3].tolist())
            training_monitor_2['err_ia_l'].append(res[4].tolist())
            training_monitor_2['alpha_ndcg'].append(res[5].tolist())
            # training_monitor['utility_l'].append(res[4][0])
            if params.with_evaluator_metrics:
                training_monitor_2['eva_sum'].append(res[-2].tolist())
                training_monitor_2['eva_ave'].append(res[-1].tolist())

            print("STEP %d  INTIAL RANKER | LOSS VALI: NULL" % step)
            # if not params.with_evaluator_metrics:
            for i, s in enumerate(params.metric_scope):
                print("@%d  MAP: %.4f  NDCG: %.4f  CLICKS: %.4f  ILAD: %.4f  ERR_IA: %.4f  ALPHA_NDCG: %.4f" % (
                    s, res[0][i], res[1][i], res[2][i], res[3][i], res[4][i], res[5][i]))
            # for i, s in enumerate(params.metric_scope):
            #     print("@%d  MAP: %.4f  NDCG: %.4f  CLICKS: %.4f  EVA_SUM: %.4f  EVA_AVE: %.4f  ALPHA_NDCG: %.4f" % (
            #         s, res[0][i], res[1][i], res[2][i], res[3][i], res[4][i], res[5][i]))
        else:
            # pass
            vali_loss, res = eval_controllable(model, test_file, params.l2_reg, params.batch_size, False,
                                               params.metric_scope)
            training_monitor['train_loss'].append(None)
            training_monitor['vali_loss'].append(None)
            training_monitor['map_l'].append(res[0][-1][-1])
            training_monitor['ndcg_l'].append(res[1][-1][-1])
            # training_monitor['de_ndcg_l'].append(res[2][0])
            training_monitor['clicks_l'].append(res[2][-1][-1])
            # training_monitor['utility_l'].append(res[4][0])
            training_monitor['ilad_l'].append(res[3][-1][-2])
            training_monitor['err_ia_l'].append(res[4][-1][-1])
            training_monitor['alpha_ndcg'].append(res[5][-1][-1])

            training_monitor_2['train_loss'].append(None)
            training_monitor_2['vali_loss'].append(None)
            training_monitor_2['map_l'].append(res[0])
            training_monitor_2['ndcg_l'].append(res[1])
            # training_monitor['de_ndcg_l'].append(res[2][0])
            training_monitor_2['clicks_l'].append(res[2])
            training_monitor_2['ilad_l'].append(res[3])
            training_monitor_2['err_ia_l'].append(res[4])
            training_monitor_2['alpha_ndcg'].append(res[5])
            # for j in [0, 5, 10]:
            for j in [0, 1, 2]:
                print("auc_prefer: ", float(j * 5) / 10)
                print("STEP %d  INTIAL RANKER | LOSS VALI: NULL" % step)
                for i, s in enumerate(params.metric_scope):
                    print("@%d  MAP: %.4f  NDCG: %.4f  CLICKS: %.4f  ILAD: %.4f  ERR_IA: %.4f  ALPHA_NDCG: %.4f" % (
                        s, res[0][j][i], res[1][j][i], res[2][j][i], res[3][j][i], res[4][j][i], res[5][j][i]))

    early_stop = False

    data = train_file
    data_size = len(data[0])
    batch_num = data_size // params.batch_size
    eval_iter_num = (data_size // 5) // params.batch_size
    print('train', data_size, batch_num)

    # begin training process
    for epoch in range(params.epoch_num):
        # if early_stop:
        #     break
        for batch_no in range(batch_num):
            data_batch = get_aggregated_batch(data, batch_size=params.batch_size, batch_no=batch_no)
            # if early_stop:
            #     break
            train_prefer = random.uniform(0, 1)
            train_prefer_step.append(train_prefer)
            if params.model_type == 'EGR_generator':

                data_batch = repeat_data(data_batch, params.rep_num)

                act_idx_out, act_probs_one, rl_sp_outputs, rl_de_outputs, mask_arr, lp_sp_data, lp_de_data, _, enc_input, \
                cate_chosen, cate_seq = model.predict(data_batch, train_prefer, params.l2_reg)


                #print(act_idx_out)
                #print(data_batch[4])
                #labels = np.array(model.build_label_reward(data_batch[4], act_idx_out))
                #auc_rewards = np.array(model.build_ndcg_reward(labels))
                #base_auc_rewards = np.array(model.build_ndcg_reward(data_batch[4]))
                #auc_rewards -= base_auc_rewards
                #print(auc_rewards)
                auc_rewards = evaluator.predict(rl_sp_outputs, rl_de_outputs, data_batch[6])
                # base_auc_rewards = evaluator.predict(np.array(data_batch[2]), np.array(data_batch[3]), data_batch[6])
                # auc_rewards = auc_rewards-base_auc_rewards
                # d_preds, d_rewards = discriminator.predict(rl_sp_outputs, rl_de_outputs, data_batch[6])
                # rewards = pred + d_rewards.reshape((-1, max_time_len)) * c_rewards_d

                _, div_rewards = model.build_erria_reward(cate_chosen, cate_seq)
                # _, base_div_rewards = model.build_erria_reward(cate_seq, cate_seq)  # rank base rerank new
                # div_rewards = div_rewards-base_div_rewards
                # rewards = pred
                # train rl-rerank
                # for _ in range(update_steps):
                loss, mean_return, auc_loss, div_loss = model.train(data_batch, rl_sp_outputs, rl_de_outputs,
                                                                    act_probs_one, act_idx_out,
                                                                    auc_rewards, div_rewards, mask_arr,
                                                                    params.c_entropy, params.lr,
                                                                    params.l2_reg,
                                                                    params.keep_prob, train_prefer=train_prefer)
                auc_train_losses_step.append(auc_loss)
                div_train_losses_step.append(div_loss)

                # train discriminator
                # if step % (update_rate_d * int(update_steps)) == 0:
                #     d_label = np.array([1] * lp_sp_data.shape[0] + [0] * rl_sp_outputs.shape[0])
                #     spar_data = np.concatenate([lp_sp_data, rl_sp_outputs], axis=0)
                #     dens_data = np.concatenate([lp_de_data, rl_de_outputs], axis=0)
                #     seq_len = np.array(data_batch[6] + data_batch[6])
                #     d_total_loss = discriminator.train([spar_data, dens_data, d_label, seq_len], lr, l2_reg)
                #     print('dis, step: %d' % (step), 'loss', d_total_loss)
            elif params.model_type == 'LAST_generator':
                training_attention_distribution, training_prediction_order, predictions, cate_seq, cate_chosen = \
                    model.rerank(data_batch, params.keep_prob, train_prefer=train_prefer)

                rl_sp_outputs, rl_de_outputs = model.build_ft_chosen(data_batch, training_prediction_order)
                #rerank_click = np.array(model.build_label_reward(data_batch[4], training_prediction_order))
                #print(training_prediction_order)
                if params.auc_rewards_type == 'iv':
                    auc_rewards = evaluator.predict(np.array(data_batch[1]), rl_sp_outputs, rl_de_outputs,
                                                    data_batch[6])
                    base_auc_rewards = evaluator.predict(np.array(data_batch[1]), np.array(data_batch[2]),
                                                         np.array(data_batch[3]), data_batch[6])
                    auc_rewards -= base_auc_rewards
                elif params.auc_rewards_type == 'label':
                    auc_rewards = np.array(model.build_label_reward(data_batch[4], training_prediction_order))
                    auc_rewards -= np.mean(auc_rewards, axis=1).reshape(-1, 1)
                elif params.auc_rewards_type == 'pv_ndcg':
                    labels = np.array(model.build_label_reward(data_batch[4], training_prediction_order))
                    auc_rewards = np.array(model.build_ndcg_reward(labels))
                    base_auc_rewards = np.array(model.build_ndcg_reward(data_batch[4]))
                    auc_rewards -= base_auc_rewards

                # _, base_div_rewards = model.build_erria_reward(cate_seq, cate_seq)  # rank base rerank new
                # _, div_rewards = model.build_erria_reward(cate_chosen, cate_seq)
                base_div_rewards = model.build_alpha_ndcg_reward(cate_seq, cate_seq)  # rank base rerank new
                div_rewards = model.build_alpha_ndcg_reward(cate_chosen, cate_seq)
                div_rewards -= base_div_rewards

                loss = model.train(data_batch, training_prediction_order, auc_rewards, div_rewards, params.lr, params.l2_reg,
                                   params.keep_prob)
            elif params.model_type == 'CMR_generator':
                training_attention_distribution, training_prediction_order, predictions, cate_seq, cate_chosen = \
                    model.rerank(data_batch, params.keep_prob, train_prefer=train_prefer)

                if params.auc_rewards_type == 'iv':
                    rl_sp_outputs, rl_de_outputs = model.build_ft_chosen(data_batch, training_prediction_order)
                    rerank_click = np.array(model.build_label_reward(data_batch[4], training_prediction_order))
                    if params.evaluator_type == 'cmr' or params.evaluator_type == 'last':
                        auc_rewards = evaluator.predict(np.array(data_batch[1]), rl_sp_outputs, rl_de_outputs,
                                                        data_batch[6])
                        base_auc_rewards = evaluator.predict(np.array(data_batch[1]), np.array(data_batch[2]),
                                                             np.array(data_batch[3]), data_batch[6])
                    elif params.evaluator_type == 'egr':
                        auc_rewards = evaluator.predict(rl_sp_outputs, rl_de_outputs, data_batch[6])
                        base_auc_rewards = evaluator.predict(np.array(data_batch[2]), np.array(data_batch[3]),
                                                             data_batch[6])

                    # base_auc_rewards = np.mean(base_auc_rewards, axis=1)
                    # base_auc_rewards = np.array(create_ave_reward(base_auc_rewards, data_batch[6]))
                    auc_rewards -= base_auc_rewards
                elif params.auc_rewards_type == 'iv2pv':
                    rl_sp_outputs, rl_de_outputs = model.build_ft_chosen(data_batch, training_prediction_order)
                    auc_rewards = evaluator.predict(np.array(data_batch[1]), rl_sp_outputs, rl_de_outputs,
                                                    data_batch[6])
                    base_auc_rewards = evaluator.predict(np.array(data_batch[1]), np.array(data_batch[2]),
                                                         np.array(data_batch[3]), data_batch[6])
                    auc_rewards = np.array(model.build_ndcg_reward(auc_rewards))
                    base_auc_rewards = np.array(model.build_ndcg_reward(base_auc_rewards))
                    auc_rewards -= base_auc_rewards
                    # auc_rewards = np.tile(auc_rewards, [1, max_time_len])
                    # better_difference = auc_rewards - base_auc_rewards  # [B,1]
                    # better_percentage = tf.where(better_difference > 0, np.ones_like(better_difference),
                    #                              tf.identity(np.zeros_like(better_difference)))  # [B,1]
                    # better_ratio = tf.divide(better_difference, auc_rewards + base_auc_rewards)  # [B,1]
                elif params.auc_rewards_type == 'label':
                    auc_rewards = np.array(model.build_label_reward(data_batch[4], training_prediction_order))
                    auc_rewards -= np.mean(auc_rewards, axis=1).reshape(-1, 1)
                elif params.auc_rewards_type == 'pv_ndcg':
                    labels = np.array(model.build_label_reward(data_batch[4], training_prediction_order))
                    auc_rewards = np.array(model.build_ndcg_reward(labels))
                    base_auc_rewards = np.array(model.build_ndcg_reward(data_batch[4]))
                    auc_rewards -= base_auc_rewards
                elif params.auc_rewards_type == 'pv_evaluator':
                    rl_sp_outputs, rl_de_outputs = model.build_ft_chosen(data_batch, training_prediction_order)
                    auc_rewards = evaluator.predict(rl_sp_outputs, rl_de_outputs, data_batch[6])
                    base_auc_rewards = evaluator.predict(np.array(data_batch[2]), np.array(data_batch[3]),
                                                         data_batch[6])
                    auc_rewards -= base_auc_rewards
                    auc_rewards = np.tile(auc_rewards, [1, max_time_len])

                if params.div_rewards_type == 'iv':
                    _, base_div_rewards = model.build_erria_reward(cate_seq, cate_seq)  # rank base rerank new
                    # base_div_rewards = np.mean(base_div_rewards, axis=1).reshape(-1, 1)    # [1 0.5 0.25] - 0.5  = [0.5 0 -0.25]
                    # base_div_rewards = np.array(create_ave_reward(base_div_rewards, data_batch[6]))
                    _, div_rewards = model.build_erria_reward(cate_chosen, cate_seq)
                    div_rewards -= base_div_rewards
                elif params.div_rewards_type == 'pv':
                    _, base_div_rewards = model.build_erria_reward(cate_seq, cate_seq)
                    base_div_rewards = np.sum(base_div_rewards, axis=1).reshape(-1, 1)
                    _, div_rewards = model.build_erria_reward(cate_chosen, cate_seq)
                    div_rewards = np.sum(div_rewards, axis=1).reshape(-1, 1)
                    div_rewards -= base_div_rewards
                    div_rewards = np.tile(div_rewards, [1, max_time_len])

                loss, auc_loss, div_loss = model.train(data_batch, training_prediction_order, auc_rewards, div_rewards,
                                                       params.lr, params.l2_reg, params.keep_prob,
                                                       train_prefer=train_prefer)
                auc_train_losses_step.append(auc_loss)
                div_train_losses_step.append(div_loss)
            elif params.model_type == 'Seq2Slate':  # [B, N, N]    [[0,0,1],[0,1,0],[1,0,0]]
                act_idx_out, act_probs_one, rl_sp_outputs, rl_de_outputs, mask_arr, lp_sp_data, lp_de_data, _, enc_input, \
                cate_chosen, cate_seq \
                    = model.predict(data_batch, train_prefer, params.l2_reg)
                mask_arr[mask_arr < 0] = 0
                div_label, _ = model.build_erria_reward(cate_chosen, cate_seq)
                loss = model.train(data_batch, rl_sp_outputs, rl_de_outputs, mask_arr, params.lr,
                                   params.l2_reg, div_label, params.keep_prob, train_prefer=train_prefer)
            else:
                loss = model.train(data_batch, params.lr, params.l2_reg, params.keep_prob, train_prefer)
            step += 1
            train_losses_step.append(loss)

            if step % eval_iter_num == 0:
                train_loss = sum(train_losses_step) / len(train_losses_step)
                training_monitor['train_loss'].append(train_loss)
                train_losses_step = []
                ave_train_prefer = sum(train_prefer_step) / len(train_prefer_step)
                training_monitor['train_prefer'].append(ave_train_prefer)
                train_prefer_step = []
                auc_train_loss = sum(auc_train_losses_step) / len(auc_train_losses_step) if len(
                    auc_train_losses_step) else 0
                training_monitor['auc_train_loss'].append(auc_train_loss)
                auc_train_losses_step = []
                div_train_loss = sum(div_train_losses_step) / len(div_train_losses_step) if len(
                    div_train_losses_step) else 0
                training_monitor['div_train_loss'].append(div_train_loss)
                div_train_losses_step = []

                if not params.controllable:
                    vali_loss, res = eval(model, test_file, params.l2_reg, params.batch_size, True,
                                          params.metric_scope, with_evaluator=params.with_evaluator_metrics,
                                          evaluator=evaluator if params.with_evaluator_metrics else None)
                    training_monitor['train_loss'].append(train_loss)
                    training_monitor['train_prefer'].append(params.acc_prefer)
                    training_monitor['auc_train_loss'].append(auc_train_loss)
                    training_monitor['div_train_loss'].append(div_train_loss)
                    training_monitor['vali_loss'].append(vali_loss)
                    training_monitor['map_l'].append(res[0][0])
                    training_monitor['ndcg_l'].append(res[1][0])
                    # training_monitor['de_ndcg_l'].append(res[2][0])
                    training_monitor['clicks_l'].append(res[2][0])
                    # training_monitor['utility_l'].append(res[4][0])
                    training_monitor['ilad_l'].append(res[3][-1])
                    training_monitor['err_ia_l'].append(res[4][-1])
                    training_monitor['alpha_ndcg'].append(res[5][-1])

                    training_monitor_2['train_loss'].append(train_loss)
                    training_monitor_2['vali_loss'].append(vali_loss)
                    training_monitor_2['map_l'].append(res[0].tolist())
                    training_monitor_2['ndcg_l'].append(res[1].tolist())
                    # training_monitor['de_ndcg_l'].append(res[2][0])
                    training_monitor_2['clicks_l'].append(res[2].tolist())
                    training_monitor_2['ilad_l'].append(res[3].tolist())
                    training_monitor_2['err_ia_l'].append(res[4].tolist())
                    training_monitor_2['alpha_ndcg'].append(res[5].tolist())
                    # training_monitor['utility_l'].append(res[4][0])

                    print("EPOCH %d STEP %d  LOSS TRAIN: %.4f | LOSS VALI: %.4f" % (epoch, step, train_loss, vali_loss))
                    # if not params.with_evaluator_metrics:
                    print("TRAIN PREFER: %.4f | AUC LOSS TRAIN: %.4f | DIV LOSS TRAIN: %.4f" % (
                        params.acc_prefer, auc_train_loss, div_train_loss))
                    for i, s in enumerate(params.metric_scope):
                        print("@%d  MAP: %.4f  NDCG: %.4f  CLICKS: %.4f  ILAD: %.4f  ERR_IA: %.4f  ALPHA_NDCG: %.4f" % (
                            s, res[0][i], res[1][i], res[2][i], res[3][i], res[4][i], res[5][i]))
                    # else:
                    #     print("TRAIN PREFER: %.4f | AUC LOSS TRAIN: %.4f | DIV LOSS TRAIN: %.4f" % (
                    #         params.acc_prefer, auc_train_loss, div_train_loss))
                    #     for i, s in enumerate(params.metric_scope):
                    #         print("@%d  MAP: %.4f  NDCG: %.4f  CLICKS: %.4f  EVA_SUM: %.4f  EVA_AVE: %.4f" % (
                    #             s, res[0][i], res[1][i], res[2][i], res[-2][i], res[-1][i]))

                    if training_monitor['map_l'][-1] >= max(training_monitor['map_l'][1:]):
                    # if training_monitor['alpha_ndcg'][-1] >= max(training_monitor['alpha_ndcg'][1:]):
                        # save model
                        model.save(save_path)
                        pkl.dump(res[-1], open(log_save_path, 'wb'))
                        print('model saved')

                    if len(training_monitor['map_l']) > 2 and epoch > 0:
                        # if (training_monitor['vali_loss'][-1] > training_monitor['vali_loss'][-2] and
                        #         training_monitor['vali_loss'][-2] > training_monitor['vali_loss'][-3]):
                        #     early_stop = True
                        if (training_monitor['map_l'][-2] - training_monitor['map_l'][-1]) <= 0.01 and (
                                training_monitor['map_l'][-3] - training_monitor['map_l'][-2]) <= 0.01:
                            early_stop = True
                else:
                    # pass
                    vali_loss, res = eval_controllable(model, test_file, params.l2_reg, params.batch_size, True,
                                                       params.metric_scope, False)
                    training_monitor['train_loss'].append(train_loss)
                    training_monitor['train_prefer'].append(ave_train_prefer)
                    training_monitor['auc_train_loss'].append(auc_train_loss)
                    training_monitor['div_train_loss'].append(div_train_loss)
                    training_monitor['vali_loss'].append(vali_loss)
                    training_monitor['map_l'].append(res[0][-1][-1])
                    training_monitor['ndcg_l'].append(res[1][-1][-1])
                    # training_monitor['de_ndcg_l'].append(res[2][0])
                    training_monitor['clicks_l'].append(res[2][-1][-1])
                    # training_monitor['utility_l'].append(res[4][0])
                    training_monitor['ilad_l'].append(res[3][-1][-2])
                    training_monitor['err_ia_l'].append(res[4][-1][-1])
                    training_monitor['alpha_ndcg'].append(res[5][-1][-1])

                    training_monitor_2['train_loss'].append(train_loss)
                    training_monitor_2['vali_loss'].append(vali_loss)
                    training_monitor_2['map_l'].append(res[0])
                    training_monitor_2['ndcg_l'].append(res[1])
                    # training_monitor['de_ndcg_l'].append(res[2][0])
                    training_monitor_2['clicks_l'].append(res[2])
                    training_monitor_2['ilad_l'].append(res[3])
                    training_monitor_2['err_ia_l'].append(res[4])
                    training_monitor_2['alpha_ndcg'].append(res[5])

                    print("EPOCH %d STEP %d  LOSS TRAIN: %.4f | LOSS VALI: %.4f" % (
                        epoch, step, train_loss, sum(vali_loss) / len(vali_loss)))
                    print("TRAIN PREFER: %.4f | AUC LOSS TRAIN: %.4f | DIV LOSS TRAIN: %.4f" % (
                        ave_train_prefer, auc_train_loss, div_train_loss))
                    # for j in [0, 5, 10]:
                    for j in [0, 1, 2]:
                        print("auc_prefer: ", float(j * 5) / 10)
                        print("STEP %d  INTIAL RANKER | LOSS VALI: NULL" % step)
                        for i, s in enumerate(params.metric_scope):
                            print("@%d  MAP: %.4f  NDCG: %.4f  CLICKS: %.4f  ILAD: %.4f  ERR_IA: %.4f  ALPHA_NDCG: %.4f" % (
                                s, res[0][j][i], res[1][j][i], res[2][j][i], res[3][j][i], res[4][j][i], res[5][j][i]))

                    if training_monitor['map_l'][-1] >= max(training_monitor['map_l'][1:]):
                        # save model
                        model.save(save_path)
                        pkl.dump(res[-1], open(log_save_path, 'wb'))
                        print('model saved')

            # generate log
            if not os.path.exists('{}/logs_{}/{}/'.format(parse.save_dir, data_set_name, max_time_len)):
                os.makedirs('{}/logs_{}/{}/'.format(parse.save_dir, data_set_name, max_time_len))
            with open('{}/logs_{}/{}/{}.monitor.pkl'.format(parse.save_dir, data_set_name, max_time_len, model_name),
                      'wb') as f:
                pkl.dump(training_monitor, f)
            with open('{}/logs_{}/{}/{}.monitor2.pkl'.format(parse.save_dir, data_set_name, max_time_len, model_name),
                      'wb') as f:
                pkl.dump(training_monitor_2, f)

        if epoch % 5 == 0 and params.controllable:
            ctl_save_path = '{}/save_model_{}/{}/{}/{}/ckpt'.format(parse.save_dir, data_set_name, max_time_len,
                                                                    model_name,
                                                                    epoch)
            model.save(ctl_save_path)
            print('model saved')


def reranker_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_time_len', default=10, type=int, help='max time length')
    parser.add_argument('--save_dir', type=str, default='./', help='dir that saves logs and model')
    parser.add_argument('--data_dir', type=str, default='./data/ad/', help='data dir')
    parser.add_argument('--model_type', default='LAST_generator',
                        choices=['PRM', 'DLCM', 'SetRank', 'GSF', 'miDNN', 'Seq2Slate', 'EGR_evaluator',
                                 'EGR_generator', 'CMR_generator', 'CMR_evaluator', 'LAST_generator',
                                 'LAST_evaluator'],
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
    parser.add_argument('--setting_path', type=str, default='./example/config/ad/last_generator_setting.json',
                        help='setting dir')
    parser.add_argument('--controllable', type=bool, default=False, help='is controllable')
    parser.add_argument('--auc_rewards_type', type=str, default='iv', help='auc rewards type')
    parser.add_argument('--div_rewards_type', type=str, default='iv', help='div rewards type')
    parser.add_argument('--evaluator_type', type=str, default='cmr', help='evaluator_type')
    parser.add_argument('--with_evaluator_metrics', type=bool, default=False, help='with_evaluator_metrics')
    parser.add_argument('--reload_model', type=bool, default=False, help='reload model')
    FLAGS, _ = parser.parse_known_args()
    return FLAGS


if __name__ == '__main__':
    # parameters
    random.seed(1237)
    set_global_determinism(1237)
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
    # train_file, val_file, test_file = pkl.load(open(os.path.join(processed_dir, 'data.data'), 'rb'))
    # props = pkl.load(open(os.path.join(processed_dir, 'prop'), 'rb'))
    # props[0] = [1e-6 for i in range(max_time_len)]
    # profile = pkl.load(open(os.path.join(processed_dir, 'user.profile'), 'rb'))

    # construct training files
    train_dir = os.path.join(processed_dir, initial_ranker + '.data.train')

    if os.path.isfile(train_dir):
        train_lists = pkl.load(open(train_dir, 'rb'))
    else:
        train_lists = construct_list(os.path.join(processed_dir, initial_ranker + '.rankings.train'), max_time_len)
        pkl.dump(train_lists, open(train_dir, 'wb'))

    # construct test files
    test_dir = os.path.join(processed_dir, initial_ranker + '.data.test')
    if os.path.isfile(test_dir):
        test_lists = pkl.load(open(test_dir, 'rb'))
    else:
        test_lists = construct_list(os.path.join(processed_dir, initial_ranker + '.rankings.test'), max_time_len)
        pkl.dump(test_lists, open(test_dir, 'wb'))

    train(train_lists, test_lists, num_ft, max_time_len, itm_spar_fnum, itm_dens_fnum, profile_fnum, parse)
