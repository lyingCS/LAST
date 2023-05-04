from librerank.CMR_evaluator import *

class LAST_evaluator(CMR_evaluator):
    def logits_layer(self):
        logits = layers.linear(self.final_neurons, self.max_time_len)
        self.before_sigmoid = logits
        logits = tf.sigmoid(logits)
        predictions = tf.reshape(logits, [-1, self.max_time_len])  # [B, N]
        seq_mask = tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32)
        predictions = seq_mask * predictions
        self.logits = predictions