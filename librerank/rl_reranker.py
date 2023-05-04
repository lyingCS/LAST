import datetime
from abc import ABC

import tensorflow as tf
import numpy as np
from tensorflow_core.python.framework import dtypes
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl as core_rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from librerank.reranker import BaseModel
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
import sys


class RLModel(BaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, itm_spar_num, itm_dens_num,
                 profile_num, max_norm=None, acc_prefer=1.0, is_controllable=False, sample_val=0.2, gamma=0.01,
                 rep_num=1, loss_type='ce'):
        super(RLModel, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, itm_spar_num, itm_dens_num,
                                      profile_num, max_norm, acc_prefer=acc_prefer, is_controllable=is_controllable)
        self.sample_val = sample_val
        self.gamma = gamma
        self.rep_num = rep_num
        self.loss_type = loss_type

        with self.graph.as_default():
            self._build_graph()

    def _build_graph(self):
        self.lstm_hidden_units = 32

        with tf.variable_scope("input"):
            self.train_phase = self.is_train
            self.sample_phase = tf.placeholder(tf.bool, name="sample_phase")  # True
            self.mask_in_raw = tf.placeholder(tf.float32, [None])
            self.div_label = tf.placeholder(tf.float32, [None, self.max_time_len])
            self.auc_label = tf.placeholder(tf.float32, [None, self.max_time_len])
            # self.idx_out_act = tf.placeholder(tf.int32, [None, self.max_time_len])
            self.item_input = self.item_seq
            self.item_label = self.label_ph  # [B, N]
            item_features = self.item_input

            self.item_size = self.max_time_len
            self.mask_in = tf.reshape(self.mask_in_raw, [-1, self.item_size])  # [B*N, N]

            self.enc_input = tf.reshape(item_features, [-1, self.item_size, self.ft_num])  # [B, N, ft_num]
            self.full_item_spar_fts = self.itm_spar_ph
            self.full_item_dens_fts = self.itm_dens_ph
            self.pv_item_spar_fts = tf.reshape(self.full_item_spar_fts, (-1, self.full_item_spar_fts.shape[-1]))
            self.pv_item_dens_fts = tf.reshape(self.full_item_dens_fts, (-1, self.full_item_dens_fts.shape[-1]))

            self.raw_dec_spar_input = tf.placeholder(tf.float32, [None, self.itm_spar_num])
            self.raw_dec_dens_input = tf.placeholder(tf.float32, [None, self.itm_dens_num])
            self.itm_spar_emb = tf.gather(self.emb_mtx, self.itm_spar_ph)
            self.raw_dec_input = tf.concat(
                [tf.reshape(self.itm_spar_emb, [-1, self.max_time_len, self.itm_spar_num * self.emb_dim]),
                 self.itm_dens_ph], axis=-1)
            self.dec_input = self.raw_dec_input

        with tf.variable_scope("encoder"):
            enc_input_train = tf.reshape(tf.tile(self.enc_input, (1, self.max_time_len, 1)),
                                         [-1, self.item_size, self.ft_num])  # [B*N, N, ft_num]
            enc_input = tf.cond(self.train_phase, lambda: enc_input_train, lambda: self.enc_input)  #
            if not self.is_controllable:
                self.enc_outputs = self.get_dnn(enc_input, [200, 80], [tf.nn.relu, tf.nn.relu], "enc_dnn")  # [B*N or B,
            # N, 80]
            else:
                self.enc_outputs = self.get_hyper_dnn(enc_input, [enc_input.get_shape()[-1].value,
                                                            200, 80], [tf.nn.relu, tf.nn.relu], "hyper_enc_dnn")

        with tf.variable_scope("encoder_state"):
            cell_dec = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_hidden_units)

        with tf.variable_scope("decoder"):
            # for training
            dec_input = tf.reshape(self.dec_input, [-1, self.max_time_len, self.ft_num])
            zero_input = tf.zeros_like(dec_input[:, :1, :])
            dec_input = tf.concat([zero_input, dec_input[:, :-1, :]], axis=1)

            zero_state = cell_dec.zero_state(tf.shape(dec_input)[0], tf.float32)  # [B]
            new_dec_input = dec_input
            dec_outputs_train, _ = tf.nn.dynamic_rnn(cell_dec, inputs=new_dec_input, time_major=False,
                                                     initial_state=zero_state)
            dec_outputs_train = tf.reshape(dec_outputs_train, [-1, 1, self.lstm_hidden_units])
            dec_outputs_train_tile = tf.tile(dec_outputs_train, [1, self.item_size, 1])

            x = tf.concat([self.enc_outputs, dec_outputs_train_tile], axis=-1)  # [B*N or B, N, ]
            if not self.is_controllable:
                self.act_logits_train = tf.reshape(
                    self.get_dnn(x, [200, 80, 1], [tf.nn.relu, tf.nn.relu, None], "dec_dnn"),
                    [-1, self.item_size])  # [B*N or B, N]
            else:
                self.act_logits_train = tf.reshape(self.get_hyper_dnn(x, [self.lstm_hidden_units + 80, 200, 80, 1],
                                                                      [tf.nn.relu, tf.nn.relu, None], "hyper_dec_dnn"),
                                                   [-1, self.item_size])  # [B*N or B, N]
            self.act_probs_train = tf.nn.softmax(self.act_logits_train)  # [B*N or N, N]
            self.act_probs_train_mask = tf.nn.softmax \
                (tf.add(tf.multiply(1. - self.mask_in, -1.0e9), self.act_logits_train))  # [B*N or N, N]

            # for predicting
            dec_input = tf.zeros([tf.shape(self.item_input)[0], self.ft_num])

            dec_states = cell_dec.zero_state(tf.shape(dec_input)[0], tf.float32)
            # mask_tmp = tf.ones([tf.shape(self.item_input)[0], self.item_size], dtype=tf.float32)
            mask_tmp = tf.cast(tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len), tf.float32)  # [B,N]

            # compute cate
            # tmp_range = tf.cast(tf.range(tf.shape(enc_input_train)[0], dtype=tf.int32), tf.int32)
            # N_tmp_range = tf.cast(
            #     tf.tile(tf.cast(tf.range(self.item_size, dtype=tf.int32), tf.int32), [tf.shape(self.enc_input)[0]]),
            #     tf.int32)
            # cate_dim = tf.cast(tf.ones(tf.shape(enc_input_train)[0], dtype=tf.int32), tf.int32)
            # cate_id = tf.stack([tmp_range, N_tmp_range, cate_dim], axis=1)  # [B, 3]
            # cate_id = tf.reshape(tf.gather_nd(enc_input_train, cate_id), [-1, self.item_size])

            mask_list = []
            act_idx_list = []
            act_probs_one_list = []
            act_probs_all_list = []
            next_dens_state_list = []
            next_spar_state_list = []
            div_list = []
            scores_pred = tf.zeros([tf.shape(self.item_input)[0], self.item_size], dtype=tf.float32)

            random_val = tf.random_uniform([], 0, 1)
            for k in range(self.max_time_len):
                new_dec_input = dec_input

                dec_outputs, dec_states = cell_dec(new_dec_input, dec_states)
                mask_list.append(mask_tmp)

                dec_outputs_tile = tf.tile(tf.reshape(dec_outputs, [-1, 1, dec_outputs.shape[-1]]),
                                           [1, self.item_size, 1])

                x = tf.concat([self.enc_outputs, dec_outputs_tile], axis=-1)
                if not self.is_controllable:
                    act_logits_pred = tf.reshape(
                        self.get_dnn(x, [200, 80, 1], [tf.nn.relu, tf.nn.relu, None], "dec_dnn"),
                        [-1, self.item_size])
                else:
                    act_logits_pred = tf.reshape(self.get_hyper_dnn(x, [self.lstm_hidden_units + 80, 200, 80, 1],
                                                                    [tf.nn.relu, tf.nn.relu, None], "hyper_dec_dnn"),
                                                 [-1, self.item_size])
                act_probs_mask = tf.nn.softmax(tf.add(tf.multiply(1. - mask_tmp, -1.0e9), act_logits_pred))  # [B, N]
                act_probs_mask_random = tf.nn.softmax(tf.add(tf.multiply(1. - mask_tmp, -1.0e9), mask_tmp))

                act_random = tf.reshape(tf.multinomial(tf.log(act_probs_mask_random), num_samples=1), [-1])
                act_stoc = tf.reshape(tf.multinomial(tf.log(act_probs_mask), num_samples=1), [-1])
                # act_det = tf.argmax(act_probs_mask, axis=1)
                # act_idx_out = tf.cond(self.sample_phase, lambda: act_stoc, lambda: act_det)
                act_idx_out = tf.cond(self.sample_phase, lambda: tf.cond(random_val < self.sample_val,
                                                                         lambda: act_random,
                                                                         lambda: act_stoc),
                                      lambda: act_stoc)  # [B]
                tmp_range = tf.cast(tf.range(tf.shape(self.item_input)[0], dtype=tf.int32), tf.int64)  # [B]
                idx_pair = tf.stack([tmp_range, act_idx_out], axis=1)  # [B, 2]

                idx_one_hot = tf.one_hot(act_idx_out, self.item_size)

                mask_tmp = mask_tmp - idx_one_hot  # [B, N]
                dec_input = tf.gather_nd(self.enc_input, idx_pair)  # [B, ft_num]
                next_full_spar_state = tf.gather_nd(self.full_item_spar_fts, idx_pair)  # [B, 5]
                next_full_dens_state = tf.gather_nd(self.full_item_dens_fts, idx_pair)  # [B, 1]
                act_probs_one = tf.gather_nd(act_probs_mask, idx_pair)  # [B]

                act_idx_list.append(act_idx_out)
                act_probs_one_list.append(act_probs_one)
                act_probs_all_list.append(act_probs_mask)
                next_spar_state_list.append(next_full_spar_state)
                next_dens_state_list.append(next_full_dens_state)

                scores_pred = scores_pred + tf.cast(idx_one_hot, dtype=tf.float32) * (1 - k * 0.03)

            self.mask_arr = tf.stack(mask_list, axis=1)  # [B, N, N]
            self.act_idx_out = tf.stack(act_idx_list, axis=1)  # [B, N]
            self.act_probs_one = tf.stack(act_probs_one_list, axis=1)  # [B, N]
            self.act_probs_all = tf.stack(act_probs_all_list, axis=1)  # [B, N, N]
            self.next_spar_state_out = tf.reshape(tf.stack(next_spar_state_list, axis=1),
                                                  [-1, self.full_item_spar_fts.shape[-1]])  # [B, spar_num]
            self.next_dens_state_out = tf.reshape(tf.stack(next_dens_state_list, axis=1),
                                                  [-1, self.full_item_dens_fts.shape[-1]])  # [B, dens_num]

            self.rerank_predict = tf.identity(tf.reshape(scores_pred, [-1, self.max_time_len]),
                                              'rerank_predict')  # [B, N]

            seq_mask = tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32)  # [B, N]
            self.y_pred = self.rerank_predict * seq_mask  # [B, N]

            # prepare for err_ia compute
            # idx_out = tf.cast(tf.reshape(self.act_idx_out, [-1]), tf.int32)  # [B*N]
            # tmp_range = tf.cast(tf.range(tf.shape(idx_out)[0], dtype=tf.int32), tf.int32)   # [0 - B*N-1]
            # # N_tmp_range = tf.cast(
            # #     tf.tile(tf.cast(tf.range(self.item_size, dtype=tf.int32), tf.int32), [tf.shape(self.enc_input)[0]]),
            # #     tf.int32)  # [0 - N-1, repeat B times]
            # cate_dim = tf.cast(tf.ones(tf.shape(idx_out)[0], dtype=tf.int32), tf.int32)
            # idx_pair = tf.stack([tmp_range, idx_out, cate_dim], axis=1)  # [B, 3]
            # cate_id = tf.stack([tmp_range, N_tmp_range, cate_dim], axis=1)
            tmp_idx_out = tf.cast(tf.reshape(self.act_idx_out, [-1, self.item_size, 1]), dtype=tf.int32)
            tmp_idx_range = tf.tile(tf.reshape(tf.range(0, tf.shape(tmp_idx_out)[0]), [-1, 1, 1]),
                                    [1, self.item_size, 1])
            tmp_idx_range = tf.cast(tf.concat([tmp_idx_range, tmp_idx_out], axis=2), dtype=tf.int32)
            self.cate_seq = tf.gather(self.itm_spar_ph, 1, axis=2)
            self.cate_chosen = tf.gather_nd(self.cate_seq, tmp_idx_range)
            self.cate_seq = tf.gather(self.itm_spar_ph, 1, axis=2)
            mask = tf.cast(tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len), tf.int32)  # [B,N]
            self.cate_chosen = self.cate_chosen * mask

        with tf.variable_scope("loss"):
            self._build_loss()

    def build_ndcg_reward(self, labels):
        # labels = labels.tolist()
        ndcg_list = []
        for label_list in labels:
            _dcg = 0
            _ideal_dcg, click_num = 0, 0
            for i in range(len(label_list)):
                _dcg += label_list[i] / np.log2(i + 2)
                if label_list[i]:
                    click_num += 1
                    _ideal_dcg += 1 / np.log2(click_num + 1)
            ndcg_list.append([_dcg/_ideal_dcg if _ideal_dcg != 0 else 0 for i in range(self.max_time_len)])
        return ndcg_list

    def build_erria_reward(self, cate_chosen, cate_seq):
        # self.idx_out_act = tf.placeholder(tf.float32, [None, self.max_time_len])
        # with tf.Session() as sess:
        # cate = cate.eval(session=tf.compat.v1.Session()).tolist()
        # cate_id = cate_id.eval(session=tf.compat.v1.Session()).tolist()
        cate_seq = cate_seq.tolist()
        cate_chosen = cate_chosen.tolist()
        div_list = []
        rl_div_reward_list = []
        for i in range(len(cate_chosen)):
            mp = {}
            rl_div_reward = []
            itm_chosen_num = 0
            for j in range(self.item_size):
                div = []
                for k in range(self.item_size):
                    if cate_seq[i][k] == 0:
                        div.append(0)
                    else:
                        div.append(1. / pow(2, mp.get(cate_seq[i][k], 0)))
                if cate_chosen[i][j] != 0:
                    itm_chosen_num += 1
                    rl_div_reward.append(1. / pow(2, mp.get(cate_chosen[i][j], 0)) / itm_chosen_num)
                else:
                    rl_div_reward.append(0)
                mp[cate_chosen[i][j]] = mp.setdefault(cate_chosen[i][j], 0) + 1
                div_list.append(div)
            rl_div_reward_list.append(rl_div_reward)
        div_reward = np.array(div_list)
        rl_div_reward_list = np.array(rl_div_reward_list)
        div_reward = div_reward / (div_reward.sum(axis=0) + 1e-5) + 1e-10
        return div_reward, rl_div_reward_list

    def build_alpha_ndcg_reward(self, cate_chosen, cate_seq):
        alpha = 0.5
        # self.idx_out_act = tf.placeholder(tf.float32, [None, self.max_time_len])
        # with tf.Session() as sess:
        # cate = cate.eval(session=tf.compat.v1.Session()).tolist()
        # cate_id = cate_id.eval(session=tf.compat.v1.Session()).tolist()
        cate_seq = cate_seq.tolist()
        cate_chosen = cate_chosen.tolist()
        rl_div_reward_list = []
        for i in range(len(cate_chosen)):
            mp = {}
            rl_div_reward = []
            itm_chosen_num = 0
            for j in range(self.item_size):
                if cate_chosen[i][j] != 0:
                    itm_chosen_num += 1
                    rl_div_reward.append(pow((1. - alpha), mp.get(cate_chosen[i][j], 0)) / pow(2, j + 1))
                else:
                    rl_div_reward.append(0)
                mp[cate_chosen[i][j]] = mp.setdefault(cate_chosen[i][j], 0) + 1
            rl_div_reward_list.append(rl_div_reward)
        rl_div_reward_list = np.array(rl_div_reward_list)
        return rl_div_reward_list

    def predict(self, batch_data, train_prefer, sample_phase=False, train_phase=False):
        with self.graph.as_default():
            act_idx_out, act_probs_one, next_state_spar_out, next_state_dens_out, mask_arr, pv_item_spar_fts, \
            pv_item_dens_fts, rerank_predict, enc_input, cate_chosen, cate_seq = self.sess.run(
                [self.act_idx_out, self.act_probs_one, self.next_spar_state_out, self.next_dens_state_out,
                 self.mask_arr, self.pv_item_spar_fts, self.pv_item_dens_fts, self.y_pred, self.enc_input,
                 self.cate_chosen, self.cate_seq],
                feed_dict={
                    self.itm_spar_ph: batch_data[2],
                    self.itm_dens_ph: batch_data[3],
                    self.seq_length_ph: batch_data[6],
                    self.is_train: train_phase,
                    self.sample_phase: sample_phase,
                    self.label_ph: batch_data[4],
                    self.controllable_auc_prefer: train_prefer,
                    self.controllable_prefer_vector: [[train_prefer, 1 - train_prefer]],
                },
            )
            return act_idx_out, act_probs_one, next_state_spar_out, next_state_dens_out, mask_arr, \
                   pv_item_spar_fts, pv_item_dens_fts, rerank_predict, enc_input, cate_chosen, cate_seq

    def eval(self, batch_data, reg_lambda, eval_prefer=0, keep_prob=1, no_print=True):
        with self.graph.as_default():
            rerank_predict = self.sess.run(self.y_pred,
                                           feed_dict={
                                               self.itm_spar_ph: batch_data[2],
                                               self.itm_dens_ph: batch_data[3],
                                               self.seq_length_ph: batch_data[6],
                                               self.is_train: False,
                                               self.sample_phase: False,
                                               self.controllable_auc_prefer: eval_prefer,
                                               self.controllable_prefer_vector: [[eval_prefer, 1 - eval_prefer]],
                                               self.keep_prob: 1})
            return rerank_predict, 0

    def rank(self, batch_data, sample_phase=False, train_phase=False):
        with self.graph.as_default():
            act_idx_out = self.sess.run(self.act_idx_out,
                                        feed_dict={
                                            self.itm_spar_ph: batch_data[2],
                                            self.itm_dens_ph: batch_data[3],
                                            self.is_train: train_phase,
                                            self.sample_phase: sample_phase})
            return act_idx_out

    def get_dnn(self, x, layer_nums, layer_acts, name="dnn"):
        input_ft = x
        assert len(layer_nums) == len(layer_acts)
        with tf.variable_scope(name):
            for i, layer_num in enumerate(layer_nums):
                input_ft = tf.contrib.layers.fully_connected(
                    inputs=input_ft,
                    num_outputs=layer_num,
                    scope='layer_%d' % i,
                    activation_fn=layer_acts[i],
                    reuse=tf.AUTO_REUSE)
        return input_ft

    def get_hyper_dnn(self, x, layer_nums, layer_acts, name="hyper_dnn"):   # [200, 15, 1]
        input_ft = x
        assert len(layer_nums) == len(layer_acts) + 1
        with tf.variable_scope(name):
            for i, layer_act in enumerate(layer_acts):
                input_ft = self.build_hyper_mlp_net_scope(input_ft, layer_nums[i], layer_nums[i + 1], 'layer_%d' % i,
                                                          layer_act)
        return input_ft

    def build_hyper_mlp_net_scope(self, inp, inp_last_dim, units, scope_name, activation=tf.nn.relu):  # [B, N, ft_num]
        # prefer_vector = tf.constant([self.controllable_auc_prefer, 1 - self.controllable_auc_prefer], dtype=tf.float32)
        w_output_dim = inp_last_dim * units
        hyper_w = tf.reshape(tf.contrib.layers.fully_connected(
            inputs=self.controllable_prefer_vector,
            num_outputs=w_output_dim,
            scope=scope_name + '_w',
            activation_fn=None,
            reuse=tf.AUTO_REUSE), [-1, units])
        hyper_b = tf.contrib.layers.fully_connected(
            inputs=self.controllable_prefer_vector,
            num_outputs=units,
            scope=scope_name + '_b',
            activation_fn=None,
            reuse=tf.AUTO_REUSE)
        # self.print_loss = tf.print("prefer_vector:", self.controllable_prefer_vector, output_stream=sys.stderr)
        ret = tf.add(tf.matmul(inp, hyper_w), hyper_b)
        if activation:
            ret = activation(ret)
        return ret

    def _build_loss(self):
        raise NotImplementedError

    def train(self, *args):
        raise NotImplementedError

    def get_long_reward(self, rewards):
        long_reward = np.zeros(rewards.shape)
        val = 0
        for i in reversed(range(self.max_time_len)):
            long_reward[:, i] = self.gamma * val + rewards[:, i]
            val = long_reward[:, i]

        returns = long_reward[:, 0]
        return long_reward, returns

    def get_long_reward_no_descent(self, rewards):
        long_reward = np.zeros(rewards.shape)
        val = 0
        for i in reversed(range(self.max_time_len)):
            long_reward[:, i] = val + rewards[:, i]
            val = long_reward[:, i]

        returns = long_reward[:, 0]
        return long_reward, returns

    def build_label_reward(self, label, action):
        ret = []
        for i in range(len(label)):
            sorted_label = []
            for j in action[i]:
                sorted_label.append(label[i][j])
            ret.append(sorted_label)
        return ret


class PPOModel(RLModel):
    def _build_loss(self):
        self.clip_value = 0.1

        with tf.variable_scope("train_input"):
            self.old_act_prob = tf.placeholder(dtype=tf.float32, shape=[None], name='old_act_prob')
            self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
            self.gaes = tf.placeholder(dtype=tf.float32, shape=[None], name='gaes')
            self.returns = tf.placeholder(dtype=tf.float32, shape=[None], name='returns')
            self.c_entropy = tf.placeholder(dtype=tf.float32, name='c_entropy')
            if True:
                self.div_gaes = tf.placeholder(dtype=tf.float32, shape=[None], name='div_gaes')
                self.div_returns = tf.placeholder(dtype=tf.float32, shape=[None], name='div_returns')
                self.auc_gaes = tf.placeholder(dtype=tf.float32, shape=[None], name='auc_gaes')
                self.auc_returns = tf.placeholder(dtype=tf.float32, shape=[None], name='auc_returns')

        act_idx_one_hot = tf.one_hot(indices=self.actions, depth=self.item_size),
        cur_act_prob = tf.reduce_sum(self.act_probs_train_mask * act_idx_one_hot, axis=-1)
        ratios = tf.exp(tf.log(tf.clip_by_value(cur_act_prob, 1e-10, 1.0))
                        - tf.log(tf.clip_by_value(self.old_act_prob, 1e-10, 1.0)))
        self.ratio = ratios
        clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - self.clip_value,
                                          clip_value_max=1 + self.clip_value)
        if False:
            loss_clip = tf.minimum(tf.multiply(self.gaes, ratios), tf.multiply(self.gaes, clipped_ratios))
        else:
            div_loss_clip = tf.minimum(tf.multiply(self.div_gaes, ratios), tf.multiply(self.div_gaes, clipped_ratios))
            auc_loss_clip = tf.minimum(tf.multiply(self.auc_gaes, ratios), tf.multiply(self.auc_gaes, clipped_ratios))
            if self.is_controllable:
                self.auc_loss = tf.multiply(auc_loss_clip, self.controllable_auc_prefer)
                self.div_loss = tf.multiply(div_loss_clip, 1 - self.controllable_auc_prefer)
                loss_clip = tf.add(tf.multiply(div_loss_clip, 1 - self.controllable_auc_prefer),
                                   tf.multiply(auc_loss_clip, self.controllable_auc_prefer))
            else:
                self.auc_loss = tf.multiply(auc_loss_clip, self.acc_prefer)
                self.div_loss = tf.multiply(div_loss_clip, 1 - self.acc_prefer)
                loss_clip = tf.add(tf.multiply(div_loss_clip, 1 - self.acc_prefer),
                                   tf.multiply(auc_loss_clip, self.acc_prefer))
        self.auc_loss = -tf.reduce_mean(self.auc_loss)
        self.div_loss = -tf.reduce_mean(self.div_loss)
        self.loss_clip = -tf.reduce_mean(loss_clip)
        self.mean_gaes = -tf.reduce_mean(self.gaes)

        # construct computation graph for loss of entropy bonus
        entropy = -tf.reduce_sum(self.act_probs_train_mask *
                                 tf.log(tf.clip_by_value(self.act_probs_train_mask, 1e-10, 1.0)), axis=-1)
        self.entropy = tf.reduce_mean(entropy)  # mean of entropy of pi(obs)

        # construct computation graph for loss
        self.loss = self.loss_clip - self.c_entropy * self.entropy

        self.g = tf.reduce_mean(self.returns)
        self.opt()

    def train(self, batch_data, raw_dec_spar_input, raw_dec_dens_input, old_act_prob, actions, auc_rewards, div_rewards,
              act_mask, c_entropy, lr, reg_lambda, keep_prop=0.8, train_prefer=1):
        with self.graph.as_default():
            gaes, returns = self.get_mix_gaes(auc_rewards, div_rewards, train_prefer)
            div_gaes, div_returns = self.get_gaes(div_rewards, False)
            auc_gaes, auc_returns = self.get_gaes(auc_rewards, True)
            # raw_dec_spar_input = raw_dec_spar_input.reshape([-1, self.ft_num])
            # old_act_prob = old_act_prob.reshape([-1])
            # actions = actions.reshape([-1])
            # train
            _, total_loss, mean_return, auc_loss, div_loss = self.sess.run(
                [self.train_step, self.loss, self.g, self.auc_loss, self.div_loss],
                feed_dict={
                    self.itm_spar_ph: batch_data[2],
                    self.itm_dens_ph: batch_data[3],
                    self.raw_dec_spar_input: raw_dec_spar_input.reshape([-1, self.itm_spar_num]),
                    self.raw_dec_dens_input: raw_dec_dens_input.reshape([-1, self.itm_dens_num]),
                    self.old_act_prob: old_act_prob.reshape([-1]),
                    self.actions: actions.reshape([-1]),
                    self.mask_in_raw: act_mask.reshape([-1]),
                    self.gaes: gaes,
                    self.returns: returns,
                    self.div_gaes: div_gaes,
                    self.div_returns: div_returns,
                    self.auc_gaes: auc_gaes,
                    self.auc_returns: auc_returns,
                    self.c_entropy: c_entropy,
                    self.reg_lambda: reg_lambda,
                    self.lr: lr,
                    self.keep_prob: keep_prop,
                    self.is_train: True,
                    self.controllable_auc_prefer: train_prefer,
                    self.controllable_prefer_vector: [[train_prefer, 1 - train_prefer]],
                })
            return total_loss, mean_return, auc_loss, div_loss

    # def test(self, sess, batch_data, raw_dec_input, old_act_prob, actions, rewards, act_mask, c_entropy):
    #         gaes, returns = self.get_gaes(rewards)
    #         raw_dec_input = raw_dec_input.reshape([-1, self.ft_num])
    #         old_act_prob = old_act_prob.reshape([-1])
    #         actions = actions.reshape([-1])
    #         # test
    #         total_loss, mean_return = sess.run(
    #             [self.loss, self.g],
    #             feed_dict={
    #                         self.itm_spar_ph: batch_data[2],
    #                         self.itm_dens_ph: batch_data[3],
    #                         self.raw_dec_input: raw_dec_input,
    #                         self.old_act_prob: old_act_prob,
    #                         self.actions: actions,
    #                         self.mask_in_raw: act_mask.reshape([-1]),
    #                         self.gaes: gaes,
    #                         self.returns: returns,
    #                         self.c_entropy: c_entropy,
    #                         self.train_phase: True
    #                         })
    #         return total_loss, mean_return

    def get_mix_gaes(self, auc_rewards, div_rewards, train_prefer):
        div_long_reward, div_returns = self.get_long_reward_no_descent(div_rewards)
        auc_long_reward, auc_returns = self.get_long_reward(auc_rewards)
        # long_reward = tf.add(tf.multiply(1-self.acc_prefer, div_long_reward), tf.multiply(self.acc_prefer, auc_long_reward))
        # returns = tf.add(tf.multiply(1-self.acc_prefer, div_returns), tf.multiply(self.acc_prefer, auc_returns))
        if not self.is_controllable:
            long_reward = (1 - self.acc_prefer) * div_long_reward + self.acc_prefer * auc_long_reward
            returns = (1 - self.acc_prefer) * div_returns + self.acc_prefer * auc_returns
        else:
            long_reward = (1 - train_prefer) * div_long_reward + train_prefer * auc_long_reward
            returns = (1 - train_prefer) * div_returns + train_prefer * auc_returns
        gaes = np.reshape(long_reward,
                          [-1, self.rep_num, self.max_time_len])
        gaes_std = gaes.std(axis=1, keepdims=True)
        gaes_std = np.where(gaes_std == 0, 1, gaes_std)
        gaes = (gaes - gaes.mean(axis=1, keepdims=True)) / gaes_std

        return gaes.reshape([-1]), returns.reshape([-1])

    def get_gaes(self, rewards, decent=True):
        if decent:
            long_reward, returns = self.get_long_reward(rewards)
        else:
            long_reward, returns = self.get_long_reward_no_descent(rewards)
        gaes = np.reshape(long_reward,
                          [self.rep_num, -1, self.max_time_len])
                          # [-1, self.rep_num, self.max_time_len])
        # gaes_std = gaes.std(axis=1, keepdims=True)
        # gaes_std = np.where(gaes_std == 0, 1, gaes_std)
        # gaes = (gaes - gaes.mean(axis=1, keepdims=True)) / gaes_std
        gaes_std = gaes.std(axis=0, keepdims=True)
        gaes_std = np.where(gaes_std == 0, 1, gaes_std)
        gaes = (gaes - gaes.mean(axis=0, keepdims=True)) / gaes_std

        return gaes.reshape([-1]), returns.reshape([-1])


class SLModel(RLModel):
    # def get_erria_reward(self, idx_out_act):
    #     with self.graph.as_default():
    #         # raw_dec_input = raw_dec_input.reshape([-1, self.ft_num])
    #         # act_mask = act_mask.reshape([-1])
    #
    #         # _, total_loss, _ = self.sess.run(
    #         #     [self.train_step, self.loss, self.print_loss],
    #         r = self.sess.run(
    #             self.div_reward,
    #             feed_dict={
    #                 self.idx_out_act: idx_out_act
    #             })
    #         return r

    def _build_loss(self):
        # self.lr = 1e-4
        self.gamma = 1
        discount = 1.0 / np.log2(np.arange(2, self.max_time_len + 2)).reshape((-1, self.max_time_len))  # [1, N]
        if self.loss_type == 'ce':
            prob_mask = self.act_probs_train_mask  # [B*N, N]
            label_auc = tf.reshape(tf.tile(self.item_label, (1, self.max_time_len)), [-1, self.item_size])  # [B,
            # N]->[B,N*N]->[B*N, N]
            if not self.is_controllable:
                label_t = tf.add(tf.multiply(self.div_label, 1 - self.acc_prefer),
                                 tf.multiply(self.acc_prefer, label_auc))
            else:
                label_t = tf.add(tf.multiply(self.div_label, 1 - self.controllable_auc_prefer),
                                 tf.multiply(self.controllable_auc_prefer, label_auc))
            # self.print_loss = tf.print("label_t: ", tf.shape(label_t))
            # label_mask = self.mask_in * label_t
            # label_mask = label_mask / tf.clip_by_value(tf.reduce_sum(label_mask, -1, keep_dims=True), 1.0, 100.0)
            label_mask = tf.nn.softmax(tf.add(tf.multiply(1.0 - self.mask_in, -1.0e9), label_t))
            # label_mask = tf.multiply(self.mask_in, label_t)
            # self.label_mask = label_mask
            # label: [1,0,1,1,0]   mask: [0,0,0,1,1]->[-max+1,-max,-max+1,0+1,0+0]->[0,0,0,0.7,0.3] label_mask [0,0,0,0.5,0.5]
            # self.kce = label_mask * tf.log(tf.clip_by_value(prob_mask, 1e-9, 1.0))
            ce = -1 * tf.reduce_mean(label_mask * tf.log(tf.clip_by_value(prob_mask, 1e-9, 1.0)), axis=-1)
            ce = tf.reshape(ce, (-1, self.max_time_len))  # [B, N]
            ce = tf.multiply(ce, discount)
            # self.print_loss = tf.print("label_t: ", tf.shape(label_t), "mask_in: ", tf.shape(self.mask_in),
            #                            "prob_mask: ", tf.shape(prob_mask), "ce: ", tf.shape(ce), output_stream=sys.stderr)
            self.loss = tf.reduce_mean(tf.reduce_sum(ce, axis=1))
        elif self.loss_type == 'hinge':
            logtis = self.act_logits_train
            label_t = tf.reshape(tf.tile(self.item_label, (1, self.max_time_len)), [-1, self.item_size])
            mask_1, mask_0 = label_t, 1 - label_t
            min_label_1 = tf.reduce_min(logtis + (1 - mask_1) * 1.0e9 + (1. - self.mask_in) * 1.0e9, -1)
            max_label_0 = tf.reduce_max(logtis + (1 - mask_0) * -1.0e9 + (1. - self.mask_in) * -1.0e9, -1)
            hg = tf.maximum(0.0, 1 - min_label_1 + max_label_0)
            hg = tf.reshape(hg, (-1, self.max_time_len))
            hg = tf.multiply(hg, discount)
            self.loss = tf.reduce_mean(tf.reduce_sum(hg, axis=1))
        else:
            raise ValueError('No loss.')

        entropy = -tf.reduce_sum(self.act_probs_train_mask *
                                 tf.log(tf.clip_by_value(self.act_probs_train_mask, 1e-10, 1.0)), axis=-1)
        self.entropy = tf.reduce_mean(entropy)  # mean of entropy of pi(obs)

        self.opt()

    def train(self, batch_data, raw_dec_spar_input, raw_dec_dens_input, act_mask, lr, reg_lambda,
              div_label, keep_prob=0.8, train_prefer=1):
        with self.graph.as_default():
            # raw_dec_input = raw_dec_input.reshape([-1, self.ft_num])
            # act_mask = act_mask.reshape([-1])

            # _, total_loss, _ = self.sess.run(
            #     [self.train_step, self.loss, self.print_loss],
            _, total_loss = self.sess.run(
                [self.train_step, self.loss],
                feed_dict={
                    self.itm_spar_ph: batch_data[2],
                    self.itm_dens_ph: batch_data[3],
                    self.label_ph: batch_data[4],
                    self.raw_dec_spar_input: raw_dec_spar_input.reshape([-1, self.itm_spar_num]),  # [B, spar_num]
                    self.raw_dec_dens_input: raw_dec_dens_input.reshape([-1, self.itm_dens_num]),  # [B, spar_num]
                    self.mask_in_raw: act_mask.reshape([-1]),  # [B*N*N]
                    self.lr: lr,
                    self.reg_lambda: reg_lambda,
                    self.keep_prob: keep_prob,
                    self.is_train: True,
                    self.controllable_auc_prefer: train_prefer,
                    self.controllable_prefer_vector: [[train_prefer, 1 - train_prefer]],
                    self.div_label: div_label,
                })
            return total_loss

    # def test(self, sess, batch_data, raw_dec_input, act_mask):
    #     raw_dec_input = raw_dec_input.reshape([-1, self.ft_num])
    #     act_mask = act_mask.reshape([-1])
    #
    #     total_loss = sess.run(
    #         [self.loss],
    #         feed_dict={
    #             self.itm_spar_ph: batch_data[2],
    #             self.itm_dens_ph: batch_data[3],
    #             self.label_ph: batch_data[4],
    #             self.raw_dec_input: raw_dec_input,
    #             self.mask_in_raw: act_mask,
    #             self.train_phase: True
    #         })
    #     return total_loss
