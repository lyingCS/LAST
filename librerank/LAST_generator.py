from librerank.CMR_generator import *
from librerank.CMR_evaluator import *

class LAST_generator(CMR_generator):

    def _build_graph(self):
        # self.acc_prefer = 1
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

            self.itm_enc_input = tf.reshape(item_features, [-1, self.item_size, self.ft_num])  # [B, N, ft_num]
            self.usr_enc_input = tf.reshape(self.usr_seq, [-1, 1, self.profile_num * self.emb_dim])
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
            # self.batch_size = tf.shape(self.dec_input)[0]
            self.batch_size = self.dec_input.get_shape()[0].value
            self.N = self.item_size
            self.use_masking = True
            self.training_sample_manner = 'sample'
            self.sample_manner = 'greedy'
            self.pv_size = self.N
            self.attention_head_nums = 2
            self.feed_context_vector = True
            self.feed_train_order = tf.placeholder(tf.bool)
            self.feed_inference_order = tf.placeholder(tf.bool)
            self.name = 'CMR_generator'
            self.train_order = tf.placeholder(tf.int64, [None, self.item_size])
            self.inference_order = tf.placeholder(tf.int64, [None, self.item_size])

        self.feature_augmentation()

        self.deep_set_encode()

        self.decoder_cell = tf.nn.rnn_cell.GRUCell(self.lstm_hidden_units)

        self.rnn_decode()

        self.save_important_variables()

        with tf.variable_scope("loss"):
            self._build_loss()

        self.add_instant_learning_channels()

    def save_important_variables(self):
        self.decoder_inputs_record = self.decoder_inputs
        self.final_state_record = self.final_state
        self.encoder_states_record = self.encoder_states
        self.decoder_cell_record = self.decoder_cell

    def _build_loss(self):
        self.gamma = 1
        if self.loss_type == 'ce':
            gamma = 0.3

            reinforce_weight = tf.range(self.pv_size, dtype=tf.float32)
            reinforce_weight = tf.reshape(reinforce_weight, [-1, 1])  # [N,1]
            reinforce_weight = tf.tile(reinforce_weight, [1, self.pv_size])  # [N,N]
            reinforce_weight = reinforce_weight - tf.transpose(reinforce_weight)  # [N,N]
            reinforce_weight = tf.where(reinforce_weight >= 0, tf.pow(gamma, reinforce_weight),
                                        tf.zeros_like(reinforce_weight))  # [N,N]

            logits = tf.stack(self.training_attention_distribution[:-1], axis=1)  # [B,10,20]
            labels = tf.one_hot(self.training_prediction_order, self.item_size)  # [B,10,20]

            self.div_label = tf.matmul(self.div_label, reinforce_weight)  # [B,N]
            div_label = tf.reshape(self.div_label, [-1, 1])
            div_ce = tf.multiply(div_label, tf.reshape(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                                               labels=labels), [-1, 1]))

            self.auc_label = tf.matmul(self.auc_label, reinforce_weight)  # [B,N]
            auc_label = tf.reshape(self.auc_label, [-1, 1])
            auc_ce = tf.multiply(auc_label, tf.reshape(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                                               labels=labels), [-1, 1]))
            ce = tf.add(tf.multiply(div_ce, 1 - self.acc_prefer),
                        tf.multiply(auc_ce, self.acc_prefer))
            ce = tf.reshape(ce, (-1, self.max_time_len))  # [B, N]

            self.loss = tf.reduce_mean(tf.reduce_sum(ce, axis=1))
        else:
            raise ValueError('No loss.')

        self.opt()

    def add_instant_learning_channels(self):
        self.extra_predictions = []
        self.extra_prediction_orders = []

        # record core variables
        # inference_prediction_order_record = self.inference_prediction_order

        # calculate gradient
        gradients = tf.gradients(self.loss, [self.final_state, self.encoder_states])
        final_state_gradient = gradients[0]  # [B, N, E]
        encoder_states_gradient = gradients[1]  # [B, N, E]

        # adaptive learning rate
        final_state_norm = tf.norm(self.final_state, axis=1, keep_dims=True)
        final_state_gradient_norm = tf.norm(final_state_gradient, axis=1, keep_dims=True)
        final_state_gradient = final_state_gradient / (
                final_state_gradient_norm + self.epsilon) * final_state_norm * 0.01  #
        self.final_state_norm = tf.norm(final_state_norm)
        self.final_state_gradient_norm = tf.norm(final_state_gradient)

        encoder_states_norm = tf.norm(self.encoder_states, axis=[1, 2], keep_dims=True)
        encoder_states_gradient_norm = tf.norm(encoder_states_gradient, axis=[1, 2], keep_dims=True)
        encoder_states_gradient = encoder_states_gradient / (
                encoder_states_gradient_norm + self.epsilon) * encoder_states_norm * 0.01

        # CMR, Seq, EGR, CMR+il, midnn, PRM; ndcg, map, avg(evaluator), sum(evaluator)

        # extra predictions
        sample_manner_list = ["Thompson_sampling"]
        #sample_manner_list = ["greedy"]
        #step_sizes = [12]
        
        num = 5
        low = 6
        high = 10
        inter = (high-low)/(num-1)

        step_sizes = list(np.arange(low, high, inter))
        simple_sampling_number = len(step_sizes)
        step_sizes = [0] * simple_sampling_number + step_sizes
        #step_sizes = [8] * simple_sampling_number + step_sizes
        self.step_sizes = step_sizes
        for sample_manner in sample_manner_list:
            for step_size in step_sizes:
                self.sample_manner = sample_manner
                sampling_function = self.get_sampling_function()  # have global state inside, need to be init everytime when used.
                # if step_size==0.0:
                #     self.print_loss2 = tf.print("i",step_size,
                #                                #  "\ndi", self.decoder_inputs_record,
                #                                # "\nfs", self.final_state_record,
                #                                # "\nes", self.encoder_states_record,
                #                                "dc",self.decoder_cell,
                #                                output_stream=sys.stderr)
                with tf.variable_scope("decoder", reuse=True):
                    inference_attention_distribution, _, extra_prediction = self.attention_based_decoder(
                        self.decoder_inputs_record,
                        self.final_state_record + step_size * final_state_gradient,
                        # self.final_state_record,
                        self.encoder_states_record + step_size * encoder_states_gradient,  # delete
                        # self.encoder_states_record,  # delete
                        self.decoder_cell_record,
                        sampling_function=sampling_function, attention_head_nums=self.attention_head_nums,
                        feed_context_vector=self.feed_context_vector)
                self.extra_predictions.append(extra_prediction)
                self.extra_prediction_orders.append(tf.stack(self.inference_prediction_order, axis=1))

    def inference(self, batch_data):
        with self.graph.as_default():
            inference_order, inference_predict, cate_seq, cate_chosen = self.sess.run([self.inference_prediction_order_record, self.predictions, self.cate_seq, self.cate_chosen],
                                           feed_dict={
                                               self.usr_profile: np.reshape(np.array(batch_data[1]),
                                                                            [-1, self.profile_num]),
                                               self.itm_spar_ph: batch_data[2],
                                               self.itm_dens_ph: batch_data[3],
                                               self.seq_length_ph: batch_data[6],
                                               self.train_order: np.zeros_like(batch_data[4]),
                                               self.feed_inference_order: False,
                                               self.feed_train_order: False,
                                               self.is_train: False,
                                               self.sample_phase: False,
                                               self.keep_prob: 1})
            return inference_order, inference_predict, 0, cate_seq, cate_chosen

    def instant_learning(self, batch_data, auc_rewards, div_rewards, inference_order):
        with self.graph.as_default():
            predictions, orders = self.sess.run([self.extra_predictions, self.extra_prediction_orders],
                                           feed_dict={
                                               self.usr_profile: np.reshape(np.array(batch_data[1]),
                                                                            [-1, self.profile_num]),
                                               self.itm_spar_ph: batch_data[2],
                                               self.itm_dens_ph: batch_data[3],
                                               self.seq_length_ph: batch_data[6],
                                               self.auc_label: auc_rewards,
                                               self.div_label: div_rewards,
                                               self.feed_inference_order: True,
                                               self.inference_order: inference_order,
                                               self.feed_train_order: True,
                                               self.train_order: inference_order,
                                               self.is_train: False,
                                               self.sample_phase: False,
                                               self.keep_prob: 1})
            return predictions, orders

    def train(self, batch_data, train_order, auc_rewards, div_rewards, lr, reg_lambda, keep_prop=0.8):
        with self.graph.as_default():
            _, total_loss, training_attention_distribution, training_prediction_order, predictions = \
                self.sess.run(
                    [self.train_step, self.loss,
                     self.training_attention_distribution, self.training_prediction_order, self.predictions],
                    feed_dict={
                        self.usr_profile: np.reshape(np.array(batch_data[1]), [-1, self.profile_num]),
                        self.itm_spar_ph: batch_data[2],
                        self.itm_dens_ph: batch_data[3],
                        self.seq_length_ph: batch_data[6],
                        self.auc_label: auc_rewards,
                        self.div_label: div_rewards,
                        self.reg_lambda: reg_lambda,
                        self.lr: lr,
                        self.keep_prob: keep_prop,
                        self.is_train: True,
                        self.feed_train_order: True,
                        self.train_order: train_order,
                    })
            return total_loss
