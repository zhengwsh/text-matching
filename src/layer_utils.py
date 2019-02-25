# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.ops import nn_ops
import tensorflow.contrib.slim as slim
from tensorflow.python.layers import core as layers_core


def my_lstm_layer(input_reps, lstm_dim, input_lengths=None, scope_name=None, reuse=False, is_training=True,
                  dropout_rate=0.2, use_cudnn=True):
    '''
    :param inputs: [batch_size, seq_len, feature_dim]
    :param lstm_dim:
    :param scope_name:
    :param reuse:
    :param is_training:
    :param dropout_rate:
    :return:
    '''
    input_reps = dropout_layer(input_reps, dropout_rate, is_training=is_training)
    with tf.variable_scope(scope_name, reuse=reuse):
        if use_cudnn:
            inputs = tf.transpose(input_reps, [1, 0, 2])
            lstm = tf.contrib.cudnn_rnn.CudnnLSTM(1, lstm_dim, direction="bidirectional",
                                                  name="{}_cudnn_bi_lstm".format(scope_name),
                                                  dropout=dropout_rate if is_training else 0)
            outputs, _ = lstm(inputs)
            outputs = tf.transpose(outputs, [1, 0, 2])
            f_rep = outputs[:, :, 0:lstm_dim]
            b_rep = outputs[:, :, lstm_dim:2 * lstm_dim]
        else:
            # context_lstm_cell_fw = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(lstm_dim)
            # context_lstm_cell_bw = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(lstm_dim)
            context_lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(lstm_dim)
            context_lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(lstm_dim)
            if is_training:
                context_lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(context_lstm_cell_fw,
                                                                     output_keep_prob=(1 - dropout_rate))
                context_lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(context_lstm_cell_bw,
                                                                     output_keep_prob=(1 - dropout_rate))
            context_lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([context_lstm_cell_fw])
            context_lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([context_lstm_cell_bw])

            (f_rep, b_rep), _ = tf.nn.bidirectional_dynamic_rnn(
                context_lstm_cell_fw, context_lstm_cell_bw, input_reps, dtype=tf.float32,
                sequence_length=input_lengths)  # [batch_size, question_len, context_lstm_dim]
            outputs = tf.concat(axis=2, values=[f_rep, b_rep])
    return (f_rep, b_rep, outputs)


def my_gru_layer(input_reps, gru_dim, input_lengths=None, scope_name=None, reuse=False, is_training=True,
                 dropout_rate=0.2, use_cudnn=True):
    '''
    :param inputs: [batch_size, seq_len, feature_dim]
    :param gru_dim:
    :param scope_name:
    :param reuse:
    :param is_training:
    :param dropout_rate:
    :return:
    '''
    input_reps = dropout_layer(input_reps, dropout_rate, is_training=is_training)
    with tf.variable_scope(scope_name, reuse=reuse):
        if use_cudnn:
            inputs = tf.transpose(input_reps, [1, 0, 2])
            gru = tf.contrib.cudnn_rnn.CudnnGRU(1, gru_dim, direction="bidirectional",
                                                name="{}_cudnn_bi_gru".format(scope_name),
                                                dropout=dropout_rate if is_training else 0)
            outputs, _ = gru(inputs)
            outputs = tf.transpose(outputs, [1, 0, 2])
            f_rep = outputs[:, :, 0:gru_dim]
            b_rep = outputs[:, :, gru_dim:2 * gru_dim]
        else:
            # context_gru_cell_fw = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(gru_dim)
            # context_gru_cell_bw = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(gru_dim)
            context_gru_cell_fw = tf.nn.rnn_cell.GRUCell(gru_dim)
            context_gru_cell_bw = tf.nn.rnn_cell.GRUCell(gru_dim)
            if is_training:
                context_gru_cell_fw = tf.nn.rnn_cell.DropoutWrapper(context_gru_cell_fw,
                                                                    output_keep_prob=(1 - dropout_rate))
                context_gru_cell_bw = tf.nn.rnn_cell.DropoutWrapper(context_gru_cell_bw,
                                                                    output_keep_prob=(1 - dropout_rate))
            context_gru_cell_fw = tf.nn.rnn_cell.MultiRNNCell([context_gru_cell_fw])
            context_gru_cell_bw = tf.nn.rnn_cell.MultiRNNCell([context_gru_cell_bw])

            (f_rep, b_rep), _ = tf.nn.bidirectional_dynamic_rnn(
                context_gru_cell_fw, context_gru_cell_bw, input_reps, dtype=tf.float32,
                sequence_length=input_lengths)  # [batch_size, question_len, context_gru_dim]
            outputs = tf.concat(axis=2, values=[f_rep, b_rep])
    return (f_rep, b_rep, outputs)


def my_cnn_layer_1d(input_reps, input_length, input_dim, filter_sizes, num_filters, input_lengths=None, scope_name=None,
                    reuse=False, is_training=True,
                    dropout_rate=0.2, use_cudnn=True):
    '''
    :param inputs: [batch_size, seq_len, feature_dim]
    :param filter_sizes:
    :param num_filters:
    :param scope_name:
    :param reuse:
    :param is_training:
    :param dropout_rate:
    :return:
    '''
    pooled_outputs = []
    # input_reps = dropout_layer(input_reps, dropout_rate, is_training=is_training)
    num_filters = num_filters[0]
    with tf.variable_scope(scope_name, reuse=reuse):
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, input_dim, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.0, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    input_reps,
                    filter=W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    # use_cudnn_on_gpu=use_cudnn,
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, input_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        h_pool_flat = dropout_layer(h_pool_flat, dropout_rate, is_training=is_training)
    return h_pool_flat


def my_cnn_layer_2d(input_reps, input_dim, filter_size, num_filter, input_lengths=None, scope_name=None, reuse=False,
                    is_training=True,
                    dropout_rate=0.2, use_cudnn=True):
    '''
    :param inputs: [batch_size, feature_dim_x, feature_dim_y]
    :param filter_size:
    :param num_filter:
    :param scope_name:
    :param reuse:
    :param is_training:
    :param dropout_rate:
    :return:
    '''
    # input_reps = dropout_layer(input_reps, dropout_rate, is_training=is_training)
    with tf.variable_scope(scope_name, reuse=reuse):
        # Convolution Layer
        filter_shape = [filter_size, filter_size, 1, num_filter]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[num_filter]), name="b")
        conv = tf.nn.conv2d(
            input_reps,
            filter=W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            # use_cudnn_on_gpu=use_cudnn,
            name="conv")
        # Apply nonlinearity
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        # Maxpooling over the outputs
        pooled = tf.nn.max_pool(
            h,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='VALID',
            name="pool")
        dim = int((input_dim - filter_size + 1) / 2)
    return pooled, dim


def my_rcnn_layer(input_reps, word_emb_dim, word_context_dim, fc_dim, input_lengths=None, scope_name=None, reuse=False,
                  is_training=True,
                  dropout_rate=0.2, use_cudnn=True):
    '''
    :param inputs: [batch_size, seq_len, feature_dim]
    :param word_emb_dim:
    :param word_context_dim:
    :param scope_name:
    :param reuse:
    :param is_training:
    :param dropout_rate:
    :return:
    '''
    with tf.variable_scope(scope_name, reuse=reuse):
        (context_representation_fw, context_representation_bw, _) = my_lstm_layer(input_reps, word_context_dim,
                                                                                  input_lengths=input_lengths,
                                                                                  scope_name=scope_name, reuse=reuse,
                                                                                  is_training=is_training,
                                                                                  dropout_rate=dropout_rate,
                                                                                  use_cudnn=use_cudnn)

        shape = [tf.shape(context_representation_fw)[0], 1, tf.shape(context_representation_fw)[2]]
        context_left = tf.concat([tf.zeros(shape), context_representation_fw[:, :-1]], axis=1, name="context_left")
        context_right = tf.concat([context_representation_bw[:, 1:], tf.zeros(shape)], axis=1, name="context_right")

        representations = tf.concat([context_left, input_reps, context_right], axis=2)

        # embedding_size = 2*word_context_dim + word_emb_dim
        # W = tf.Variable(tf.random_uniform([embedding_size, fc_dim], -1.0, 1.0), name="W")
        # b = tf.Variable(tf.constant(0.1, shape=[fc_dim]), name="b")
        # representations = tf.einsum('aij,jk->aik', representations, W) + b

        return representations


def my_rcnn_layer2(input_reps, batch_size, word_emb_dim, word_context_dim, input_lengths=None, scope_name=None,
                   reuse=False, is_training=True,
                   dropout_rate=0.2, use_cudnn=True):
    '''
    :param inputs: [batch_size, seq_len, feature_dim]
    :param word_emb_dim:
    :param word_context_dim:
    :param scope_name:
    :param reuse:
    :param is_training:
    :param dropout_rate:
    :return:
    '''
    import copy
    with tf.variable_scope(scope_name, reuse=reuse):
        # define weights here
        initializer = tf.random_normal_initializer(stddev=0.1)
        left_side_first_word = tf.get_variable("left_side_first_word", shape=[batch_size, word_emb_dim],
                                               initializer=initializer)
        right_side_last_word = tf.get_variable("right_side_last_word", shape=[batch_size, word_emb_dim],
                                               initializer=initializer)
        W_l = tf.get_variable("W_l", shape=[word_context_dim, word_context_dim], initializer=initializer)
        W_r = tf.get_variable("W_r", shape=[word_context_dim, word_context_dim], initializer=initializer)
        W_sl = tf.get_variable("W_sl", shape=[word_emb_dim, word_context_dim], initializer=initializer)
        W_sr = tf.get_variable("W_sr", shape=[word_emb_dim, word_context_dim], initializer=initializer)

        # rnn-cnn layer
        def get_context_left(context_left, embedding_previous):
            left_c = tf.matmul(context_left, W_l)  # context_left:[batch_size,embed_size]; W_l:[embed_size,embed_size]
            left_e = tf.matmul(embedding_previous, W_sl)  # embedding_previous; [batch_size,embed_size]
            left_h = left_c + left_e
            context_left = tf.nn.relu(left_h, name="relu")  # [batch_size,embed_size]
            return context_left

        def get_context_right(context_right, embedding_afterward):
            right_c = tf.matmul(context_right, W_r)
            right_e = tf.matmul(embedding_afterward, W_sr)
            right_h = right_c + right_e
            context_right = tf.nn.relu(right_h, name="relu")
            return context_right

        # 1. get list of context left
        embedded_words_split = tf.split(input_reps, input_lengths,
                                        axis=1)  # sentence_length * [batch_size,1,embed_size]
        embedded_words_squeezed = [tf.squeeze(x, axis=1) for x in
                                   embedded_words_split]  # sentence_length * [batch_size,embed_size]
        embedding_previous = left_side_first_word
        context_left_previous = tf.zeros((batch_size, word_context_dim))
        context_left_list = []
        for i, current_embedding_word in enumerate(
                embedded_words_squeezed):  # sentence_length * [batch_size,embed_size]
            context_left = get_context_left(context_left_previous, embedding_previous)  # [batch_size,embed_size]
            context_left_list.append(context_left)  # append result to list
            embedding_previous = current_embedding_word  # assign embedding_previous
            context_left_previous = context_left  # assign context_left_previous
        # 2. get context right
        embedded_words_squeezed2 = copy.copy(embedded_words_squeezed)
        embedded_words_squeezed2.reverse()
        embedding_afterward = right_side_last_word
        context_right_afterward = tf.zeros((batch_size, word_context_dim))
        context_right_list = []
        for j, current_embedding_word in enumerate(embedded_words_squeezed2):
            context_right = get_context_right(context_right_afterward, embedding_afterward)
            context_right_list.append(context_right)
            embedding_afterward = current_embedding_word
            context_right_afterward = context_right
        # 3.ensemble left, embedding, right to output
        output_list = []
        for index, current_embedding_word in enumerate(embedded_words_squeezed):
            representation = tf.concat([context_left_list[index], current_embedding_word, context_right_list[index]],
                                       axis=1)
            representation = dropout_layer(representation, dropout_rate, is_training=is_training)
            output_list.append(representation)  # shape:sentence_length * [batch_size,embed_size*3]
        # 4. stack list to a tensor
        outputs = tf.stack(output_list, axis=1)  # shape:[batch_size,sentence_length,embed_size*3]

        return outputs


def dropout_layer(input_reps, dropout_rate, is_training=True):
    if is_training:
        output_repr = tf.nn.dropout(input_reps, (1 - dropout_rate))
    else:
        output_repr = input_reps
    return output_repr


def cosine_distance(y1, y2, cosine_norm=True, eps=1e-6):
    # cosine_norm = True
    # y1 [....,a, 1, d]
    # y2 [....,1, b, d]
    cosine_numerator = tf.reduce_sum(tf.multiply(y1, y2), axis=-1)
    if not cosine_norm:
        return tf.tanh(cosine_numerator)
    y1_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y1), axis=-1), eps))
    y2_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y2), axis=-1), eps))
    return cosine_numerator / y1_norm / y2_norm


def euclidean_distance(y1, y2, eps=1e-6):
    distance = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y1 - y2), axis=-1), eps))
    return distance


def l1_distance(y1, y2):
    distance = tf.reduce_sum(tf.abs(tf.subtract(y1, y2)), axis=-1)
    return distance


def cross_entropy(logits, truth, mask=None):
    # logits: [batch_size, passage_len]
    # truth: [batch_size, passage_len]
    # mask: [batch_size, passage_len]
    if mask is not None: logits = tf.multiply(logits, mask)
    xdev = tf.subtract(logits, tf.expand_dims(tf.reduce_max(logits, 1), -1))
    log_predictions = tf.subtract(xdev, tf.expand_dims(tf.log(tf.reduce_sum(tf.exp(xdev), -1)), -1))
    result = tf.multiply(truth, log_predictions)  # [batch_size, passage_len]
    if mask is not None: result = tf.multiply(result, mask)  # [batch_size, passage_len]
    return tf.multiply(-1.0, tf.reduce_sum(result, -1))  # [batch_size]


def projection_layer(in_val, input_size, output_size, activation_func=tf.tanh, scope=None):
    # in_val: [batch_size, passage_len, dim]
    input_shape = tf.shape(in_val)
    batch_size = input_shape[0]
    passage_len = input_shape[1]
    # feat_dim = input_shape[2]
    in_val = tf.reshape(in_val, [batch_size * passage_len, input_size])
    # with tf.variable_scope(scope or "projection_layer"):
    full_w = tf.get_variable("full_w", [input_size, output_size], dtype=tf.float32)
    full_b = tf.get_variable("full_b", [output_size], dtype=tf.float32)
    outputs = activation_func(tf.nn.xw_plus_b(in_val, full_w, full_b))
    outputs = tf.reshape(outputs, [batch_size, passage_len, output_size])
    return outputs  # [batch_size, passage_len, output_size]


def projection_layer2(in_val, input_size, output_size, activation_func=tf.tanh, scope=None):
    # in_val: [batch_size, dim]
    input_shape = tf.shape(in_val)
    batch_size = input_shape[0]
    # feat_dim = input_shape[1]
    with tf.variable_scope(scope or "projection_layer2"):
        full_w0 = tf.get_variable("full_w0", [input_size, int(input_size / 2)], dtype=tf.float32)
        full_b0 = tf.get_variable("full_b0", [int(input_size / 2)], dtype=tf.float32)
        logits = tf.matmul(in_val, full_w0) + full_b0
        full_w = tf.get_variable("full_w", [int(input_size / 2), output_size], dtype=tf.float32)
        full_b = tf.get_variable("full_b", [output_size], dtype=tf.float32)
        outputs = activation_func(tf.nn.xw_plus_b(logits, full_w, full_b))
    return outputs, output_size  # [batch_size, output_size]


def highway_layer(in_val, output_size, activation_func=tf.tanh, scope=None):
    # in_val: [batch_size, passage_len, dim]
    input_shape = tf.shape(in_val)
    batch_size = input_shape[0]
    passage_len = input_shape[1]
    # feat_dim = input_shape[2]
    in_val = tf.reshape(in_val, [batch_size * passage_len, output_size])
    with tf.variable_scope(scope or "highway_layer"):
        highway_w = tf.get_variable("highway_w", [output_size, output_size], dtype=tf.float32)
        highway_b = tf.get_variable("highway_b", [output_size], dtype=tf.float32)
        full_w = tf.get_variable("full_w", [output_size, output_size], dtype=tf.float32)
        full_b = tf.get_variable("full_b", [output_size], dtype=tf.float32)
        trans = activation_func(tf.nn.xw_plus_b(in_val, full_w, full_b))
        gate = tf.nn.sigmoid(tf.nn.xw_plus_b(in_val, highway_w, highway_b))
        outputs = tf.add(tf.multiply(trans, gate), tf.multiply(in_val, tf.subtract(1.0, gate)), "y")
    outputs = tf.reshape(outputs, [batch_size, passage_len, output_size])
    return outputs


def multi_highway_layer(in_val, output_size, num_layers, activation_func=tf.tanh, scope_name=None, reuse=False):
    with tf.variable_scope(scope_name, reuse=reuse):
        for i in range(num_layers):
            cur_scope_name = scope_name + "-{}".format(i)
            in_val = highway_layer(in_val, output_size, activation_func=activation_func, scope=cur_scope_name)
    return in_val


def collect_representation(representation, positions):
    # representation: [batch_size, node_num, feature_dim]
    # positions: [batch_size, neigh_num]
    return collect_probs(representation, positions)


def collect_final_step_of_lstm(lstm_representation, lengths):
    # lstm_representation: [batch_size, passsage_length, dim]
    # lengths: [batch_size]
    lengths = tf.maximum(lengths, tf.zeros_like(lengths, dtype=tf.int32))

    batch_size = tf.shape(lengths)[0]
    batch_nums = tf.range(0, limit=batch_size)  # shape (batch_size)
    indices = tf.stack((batch_nums, lengths), axis=1)  # shape (batch_size, 2)
    result = tf.gather_nd(lstm_representation, indices, name='last-forwar-lstm')
    return result  # [batch_size, dim]


def collect_mean_step_of_lstm(lstm_representation, lengths=None):
    """
    Given an input tensor (e.g., the outputs of a LSTM), do mean pooling
    over the last dimension of the input.

    For example, if the input was the output of a LSTM of shape
    (batch_size, sequence length, hidden_dim), this would
    calculate a mean pooling over the last dimension (taking the padding
    into account, if provided) to output a tensor of shape
    (batch_size, hidden_dim).

    Parameters
    ----------
    lstm_representation: Tensor
        An input tensor, preferably the output of a tensorflow RNN.
        The mean-pooled representation of this output will be calculated
        over the last dimension.
    lengths: Tensor, optional (default=None)
        A tensor of dimension (batch_size, ) indicating the length
        of the sequences before padding was applied.

    Returns
    -------
    mean_pooled_output: Tensor
        A tensor of one less dimension than the input, with the size of the
        last dimension equal to the hidden dimension state size.
    """
    # shape (batch_size, lengths)
    lstm_representation_sum = tf.reduce_sum(lstm_representation, axis=-2)
    if lengths is None:
        lengths = tf.shape(lstm_representation)[-2]
    # Expand sequence length from shape (batch_size,) to
    # (batch_size, 1) for broadcasting to work.
    expanded_sequence_length = tf.cast(tf.expand_dims(lengths, -1),
                                       "float32") + 1e-08
    # Now, divide by the length of each sequence.
    # shape (batch_size, lengths)
    mean_pooled_input = lstm_representation_sum / expanded_sequence_length
    return mean_pooled_input  # [batch_size, dim]


def collect_probs(probs, positions):
    # probs [batch_size, chunks_size]
    # positions [batch_size, pair_size]
    batch_size = tf.shape(probs)[0]
    pair_size = tf.shape(positions)[1]
    batch_nums = tf.range(0, limit=batch_size)  # shape (batch_size)
    batch_nums = tf.reshape(batch_nums, shape=[-1, 1])  # [batch_size, 1]
    batch_nums = tf.tile(batch_nums, multiples=[1, pair_size])  # [batch_size, pair_size]

    indices = tf.stack((batch_nums, positions), axis=2)  # shape (batch_size, pair_size, 2)
    pair_probs = tf.gather_nd(probs, indices)
    # pair_probs = tf.reshape(pair_probs, shape=[batch_size, pair_size])
    return pair_probs


def calcuate_attention(in_value_1, in_value_2, feature_dim1, feature_dim2, scope_name='att',
                       att_type='symmetric', att_dim=20, remove_diagnoal=False, mask1=None, mask2=None,
                       is_training=False, dropout_rate=0.2):
    input_shape = tf.shape(in_value_1)
    batch_size = input_shape[0]
    len_1 = input_shape[1]
    len_2 = tf.shape(in_value_2)[1]

    in_value_1 = dropout_layer(in_value_1, dropout_rate, is_training=is_training)
    in_value_2 = dropout_layer(in_value_2, dropout_rate, is_training=is_training)
    with tf.variable_scope(scope_name):
        # calculate attention ==> a: [batch_size, len_1, len_2]
        atten_w1 = tf.get_variable("atten_w1", [feature_dim1, att_dim], dtype=tf.float32)
        if feature_dim1 == feature_dim2:
            atten_w2 = atten_w1
        else:
            atten_w2 = tf.get_variable("atten_w2", [feature_dim2, att_dim], dtype=tf.float32)
        atten_value_1 = tf.matmul(tf.reshape(in_value_1, [batch_size * len_1, feature_dim1]),
                                  atten_w1)  # [batch_size*len_1, feature_dim]
        atten_value_1 = tf.reshape(atten_value_1, [batch_size, len_1, att_dim])
        atten_value_2 = tf.matmul(tf.reshape(in_value_2, [batch_size * len_2, feature_dim2]),
                                  atten_w2)  # [batch_size*len_2, feature_dim]
        atten_value_2 = tf.reshape(atten_value_2, [batch_size, len_2, att_dim])

        if att_type == 'additive':
            atten_b = tf.get_variable("atten_b", [att_dim], dtype=tf.float32)
            atten_v = tf.get_variable("atten_v", [1, att_dim], dtype=tf.float32)
            atten_value_1 = tf.expand_dims(atten_value_1, axis=2,
                                           name="atten_value_1")  # [batch_size, len_1, 'x', feature_dim]
            atten_value_2 = tf.expand_dims(atten_value_2, axis=1,
                                           name="atten_value_2")  # [batch_size, 'x', len_2, feature_dim]
            atten_value = atten_value_1 + atten_value_2  # + tf.expand_dims(tf.expand_dims(tf.expand_dims(atten_b, axis=0), axis=0), axis=0)
            atten_value = nn_ops.bias_add(atten_value, atten_b)
            atten_value = tf.tanh(atten_value)  # [batch_size, len_1, len_2, feature_dim]
            atten_value = tf.reshape(atten_value, [-1,
                                                   att_dim]) * atten_v  # tf.expand_dims(atten_v, axis=0) # [batch_size*len_1*len_2, feature_dim]
            atten_value = tf.reduce_sum(atten_value, axis=-1)
            atten_value = tf.reshape(atten_value, [batch_size, len_1, len_2])
        else:
            atten_value_1 = tf.tanh(atten_value_1)
            # atten_value_1 = tf.nn.relu(atten_value_1)
            atten_value_2 = tf.tanh(atten_value_2)
            # atten_value_2 = tf.nn.relu(atten_value_2)
            diagnoal_params = tf.get_variable("diagnoal_params", [1, 1, att_dim], dtype=tf.float32)
            atten_value_1 = atten_value_1 * diagnoal_params
            atten_value = tf.matmul(atten_value_1, atten_value_2, transpose_b=True)  # [batch_size, len_1, len_2]

        # normalize
        if remove_diagnoal:
            diagnoal = tf.ones([len_1], tf.float32)  # [len1]
            diagnoal = 1.0 - tf.diag(diagnoal)  # [len1, len1]
            diagnoal = tf.expand_dims(diagnoal, axis=0)  # ['x', len1, len1]
            atten_value = atten_value * diagnoal
        if mask1 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask1, axis=-1))
        if mask2 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask2, axis=1))
        atten_value = tf.nn.softmax(atten_value, name='atten_value')  # [batch_size, len_1, len_2]
        if remove_diagnoal: atten_value = atten_value * diagnoal
        if mask1 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask1, axis=-1))
        if mask2 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask2, axis=1))

    return atten_value


def inter_attention(in_value_1, in_value_2, feature_dim1, feature_dim2, scope_name='inter_att',
                    att_type='symmetric', att_dim=20, remove_diagnoal=False, mask1=None, mask2=None, is_training=False,
                    dropout_rate=0.2):
    input_shape = tf.shape(in_value_1)
    batch_size = input_shape[0]
    len_1 = input_shape[1]
    len_2 = tf.shape(in_value_2)[1]

    in_value_1 = dropout_layer(in_value_1, dropout_rate, is_training=is_training)
    in_value_2 = dropout_layer(in_value_2, dropout_rate, is_training=is_training)
    with tf.variable_scope(scope_name):
        # calculate attention ==> a: [batch_size, len_1, len_2]
        atten_w1 = tf.get_variable("atten_w1", [feature_dim1, att_dim], dtype=tf.float32)
        if feature_dim1 == feature_dim2:
            atten_w2 = atten_w1
        else:
            atten_w2 = tf.get_variable("atten_w2", [feature_dim2, att_dim], dtype=tf.float32)
        atten_value_1 = tf.matmul(tf.reshape(in_value_1, [batch_size * len_1, feature_dim1]),
                                  atten_w1)  # [batch_size*len_1, feature_dim]
        atten_value_1 = tf.reshape(atten_value_1, [batch_size, len_1, att_dim])
        atten_value_2 = tf.matmul(tf.reshape(in_value_2, [batch_size * len_2, feature_dim2]),
                                  atten_w2)  # [batch_size*len_2, feature_dim]
        atten_value_2 = tf.reshape(atten_value_2, [batch_size, len_2, att_dim])

        if att_type == 'additive':
            atten_b = tf.get_variable("atten_b", [att_dim], dtype=tf.float32)
            atten_v = tf.get_variable("atten_v", [1, att_dim], dtype=tf.float32)
            atten_value_1 = tf.expand_dims(atten_value_1, axis=2,
                                           name="atten_value_1")  # [batch_size, len_1, 'x', feature_dim]
            atten_value_2 = tf.expand_dims(atten_value_2, axis=1,
                                           name="atten_value_2")  # [batch_size, 'x', len_2, feature_dim]
            atten_value = atten_value_1 + atten_value_2  # + tf.expand_dims(tf.expand_dims(tf.expand_dims(atten_b, axis=0), axis=0), axis=0)
            atten_value = nn_ops.bias_add(atten_value, atten_b)
            atten_value = tf.tanh(atten_value)  # [batch_size, len_1, len_2, feature_dim]
            atten_value = tf.reshape(atten_value, [-1,
                                                   att_dim]) * atten_v  # tf.expand_dims(atten_v, axis=0) # [batch_size*len_1*len_2, feature_dim]
            atten_value = tf.reduce_sum(atten_value, axis=-1)
            atten_value = tf.reshape(atten_value, [batch_size, len_1, len_2])
        else:
            atten_value_1 = tf.tanh(atten_value_1)
            # atten_value_1 = tf.nn.relu(atten_value_1)
            atten_value_2 = tf.tanh(atten_value_2)
            # atten_value_2 = tf.nn.relu(atten_value_2)
            atten_value = tf.matmul(atten_value_1, tf.transpose(atten_value_2, [0, 2, 1]))  # [batch_size, len_1, len_2]

        # cross mask
        m1_m2 = tf.multiply(tf.expand_dims(mask1, 2), tf.expand_dims(mask2, 1))
        # compute the unnormalized attention for all word pairs
        # raw_attentions has shape (batch, mlen1, mlen2)
        raw_atten = tf.multiply(atten_value, m1_m2)
        # weighted attention,
        # using Softmax at two directions axis=-1 and axis=-2, for alpha and beta respectively
        atten1 = tf.exp(raw_atten - tf.reduce_max(raw_atten, axis=2, keepdims=True))
        atten2 = tf.exp(raw_atten - tf.reduce_max(raw_atten, axis=1, keepdims=True))
        # mask
        atten1 = tf.multiply(atten1, tf.expand_dims(mask2, 1))
        atten2 = tf.multiply(atten2, tf.expand_dims(mask1, 2))
        # get softmax value
        atten1 = tf.divide(atten1, tf.reduce_sum(atten1, axis=2, keepdims=True))
        atten2 = tf.divide(atten2, tf.reduce_sum(atten2, axis=1, keepdims=True))
        # mask
        atten1 = tf.multiply(atten1, m1_m2)
        atten2 = tf.multiply(atten2, m1_m2)
        # here (alpha, beta) = (beta, alpha) in the paper
        # represents the soft alignment in the other sentence
        att_in_value_1_contexts = tf.matmul(atten1, in_value_2, name='beta')
        att_in_value_2_contexts = tf.matmul(tf.transpose(atten2, [0, 2, 1]), in_value_1, name='alpha')

    return atten1, atten2, att_in_value_1_contexts, att_in_value_2_contexts


def intra_attention(in_value_1, feature_dim1, scope_name='intra_att',
                    att_type='symmetric', att_dim=20, remove_diagnoal=False, mask1=None, is_training=False,
                    dropout_rate=0.2):
    input_shape = tf.shape(in_value_1)
    batch_size = input_shape[0]
    len_1 = input_shape[1]

    in_value_1 = dropout_layer(in_value_1, dropout_rate, is_training=is_training)
    with tf.variable_scope(scope_name):
        # calculate attention ==> a: [batch_size, len_1, len_1]
        atten_w1 = tf.get_variable("atten_w1", [feature_dim1, att_dim], dtype=tf.float32)
        atten_value_1 = tf.matmul(tf.reshape(in_value_1, [batch_size * len_1, feature_dim1]),
                                  atten_w1)  # [batch_size*len_1, feature_dim]
        atten_value_1 = tf.reshape(atten_value_1, [batch_size, len_1, att_dim])

        if att_type == 'additive':
            atten_b = tf.get_variable("atten_b", [att_dim], dtype=tf.float32)
            atten_v = tf.get_variable("atten_v", [1, att_dim], dtype=tf.float32)
            atten_value_1 = tf.expand_dims(atten_value_1, axis=2,
                                           name="atten_value_1")  # [batch_size, len_1, 'x', feature_dim]
            atten_value_2 = tf.expand_dims(atten_value_1, axis=1,
                                           name="atten_value_2")  # [batch_size, 'x', len_1, feature_dim]
            atten_value = atten_value_1 + atten_value_2  # + tf.expand_dims(tf.expand_dims(tf.expand_dims(atten_b, axis=0), axis=0), axis=0)
            atten_value = nn_ops.bias_add(atten_value, atten_b)
            atten_value = tf.tanh(atten_value)  # [batch_size, len_1, len_2, feature_dim]
            atten_value = tf.reshape(atten_value, [-1,
                                                   att_dim]) * atten_v  # tf.expand_dims(atten_v, axis=0) # [batch_size*len_1*len_2, feature_dim]
            atten_value = tf.reduce_sum(atten_value, axis=-1)
            atten_value = tf.reshape(atten_value, [batch_size, len_1, len_1])
        else:
            atten_value_1 = tf.tanh(atten_value_1)
            # atten_value_1 = tf.nn.relu(atten_value_1)
            atten_value = tf.matmul(atten_value_1, atten_value_1, transpose_b=True)  # [batch_size, len_1, len_2]

        # normalize
        if mask1 is not None:
            atten_value = tf.multiply(atten_value, tf.expand_dims(mask1, axis=-1))
            atten_value = tf.multiply(atten_value, tf.expand_dims(mask1, axis=1))
        atten_value = tf.nn.softmax(atten_value, name='atten_value')  # [batch_size, len_1, len_2]
        if mask1 is not None:
            atten_value = tf.multiply(atten_value, tf.expand_dims(mask1, axis=-1))
            atten_value = tf.multiply(atten_value, tf.expand_dims(mask1, axis=1))

        att_in_value_1_contexts = tf.matmul(atten_value, in_value_1)

    return atten_value, att_in_value_1_contexts


def weighted_sum(atten_scores, in_values):
    '''

    :param atten_scores: # [batch_size, len1, len2]
    :param in_values: [batch_size, len2, dim]
    :return:
    '''
    return tf.matmul(atten_scores, in_values)


def cal_relevancy_matrix(in_question_repres, in_passage_repres):
    in_question_repres_tmp = tf.expand_dims(in_question_repres, 1)  # [batch_size, 1, question_len, dim]
    in_passage_repres_tmp = tf.expand_dims(in_passage_repres, 2)  # [batch_size, passage_len, 1, dim]
    relevancy_matrix = cosine_distance(in_question_repres_tmp,
                                       in_passage_repres_tmp)  # [batch_size, passage_len, question_len]
    return relevancy_matrix


def mask_relevancy_matrix(relevancy_matrix, question_mask, passage_mask):
    # relevancy_matrix: [batch_size, passage_len, question_len]
    # question_mask: [batch_size, question_len]
    # passage_mask: [batch_size, passsage_len]
    if question_mask is not None:
        relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(question_mask, 1))
    relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(passage_mask, 2))
    return relevancy_matrix


def compute_gradients(tensor, var_list):
    grads = tf.gradients(tensor, var_list)
    return [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(var_list, grads)]


############################################
########## MPCNN model ####
############################################
def build_block_A(x, filter_sizes, poolings, W1, b1, is_training):
    out = []
    with tf.name_scope("bulid_block_A"):
        for pooling in poolings:
            pools = []
            for i, ws in enumerate(filter_sizes):
                with tf.name_scope("conv-pool-%s" % ws):
                    # Convolution Layer
                    conv = tf.nn.conv2d(x, W1[i], strides=[1, 1, 1, 1], padding="VALID", use_cudnn_on_gpu=True,
                                        name='conv')
                    # Apply nonlinearity
                    conv = tf.nn.relu(tf.nn.bias_add(conv, b1[i]),
                                      name="relu")  # [batch_size, sentence_length-ws+1, 1, num_filters_A]
                    # Add batch normalization
                    # conv = tf.layers.batch_normalization(conv, axis=0)
                    # conv = slim.batch_norm(inputs=conv, activation_fn=tf.nn.tanh, is_training=is_training)
                    # Pooling over the outputs
                    pool = pooling(conv, axis=1)  # [batch_size, 1, num_filters_A]
                pools.append(pool)
            out.append(pools)
        return out  # len(poolings) * len(filter_sizes) * [batch_size， 1， num_filters_A]


def build_block_B(x, filter_sizes, poolings, W2, b2, is_training):
    out = []
    with tf.name_scope("bulid_block_B"):
        for pooling in poolings[:-1]:
            pools = []
            for i, ws in enumerate(filter_sizes[:-1]):
                with tf.name_scope("per_conv-pool-%s" % ws):
                    pool = per_dim_conv_layer(x, W2[i], b2[i], pooling, is_training)
                pools.append(pool)
            out.append(pools)
        return out  # len(poolings)-1 * len(filter_sizes)-1 * [batch_size， embed_size， num_filters_B]


def per_dim_conv_layer(x, w, b, pooling, is_training):
    '''
    per_dim convolution
    :param x: [batch_size, sentence_length, embed_size, 1]
    :param w: [filter_size, embed_size, 1, num_filters]
    :param b: [num_filters, embed_size]
    :param pooling:
    :return:
    '''
    input_unstack = tf.unstack(x, axis=2)
    w_unstack = tf.unstack(w, axis=1)
    b_unstack = tf.unstack(b, axis=1)
    convs = []
    for i in range(x.get_shape()[2]):
        conv = tf.nn.conv1d(input_unstack[i], w_unstack[i], stride=1, padding="VALID", use_cudnn_on_gpu=True,
                            name='per_conv')
        conv = tf.nn.relu(tf.nn.bias_add(conv, b_unstack[i]), name="relu")
        # conv = tf.layers.batch_normalization(conv, axis=0)
        # conv = slim.batch_norm(inputs=conv, activation_fn=tf.nn.tanh, is_training=is_training)
        convs.append(conv)
    conv = tf.stack(convs, axis=2)  # [batch_size, sentence_length-ws+1, embed_size, num_filters_B]
    pool = pooling(conv, axis=1)  # [batch_size, embed_size, num_filters_B]
    return pool


############################################
########## DIIN model ######################
############################################
def bi_attention_mx(config, is_train, p, h, p_mask=None, h_mask=None, scope=None):  # [N, L, 2d]
    with tf.variable_scope(scope or "dense_logit_bi_attention"):
        PL = p.get_shape().as_list()[1]
        HL = h.get_shape().as_list()[1]
        p_aug = tf.tile(tf.expand_dims(p, 2), [1, 1, HL, 1])
        h_aug = tf.tile(tf.expand_dims(h, 1), [1, PL, 1, 1])  # [N, PL, HL, 2d]

        if p_mask is None:
            ph_mask = None
        else:
            p_mask_aug = tf.reduce_any(tf.cast(tf.tile(tf.expand_dims(p_mask, 2), [1, 1, HL, 1]), tf.bool), axis=3)
            h_mask_aug = tf.reduce_any(tf.cast(tf.tile(tf.expand_dims(h_mask, 1), [1, PL, 1, 1]), tf.bool), axis=3)
            ph_mask = p_mask_aug & h_mask_aug
        ph_mask = None

        h_logits = p_aug * h_aug

        return h_logits


def self_attention(config, is_train, p, p_mask=None, scope=None):  # [N, L, 2d]
    with tf.variable_scope(scope or "self_attention"):
        PL = p.get_shape().as_list()[1]
        dim = p.get_shape().as_list()[-1]
        # HL = tf.shape(h)[1]
        p_aug_1 = tf.tile(tf.expand_dims(p, 2), [1, 1, PL, 1])
        p_aug_2 = tf.tile(tf.expand_dims(p, 1), [1, PL, 1, 1])  # [N, PL, HL, 2d]

        if p_mask is None:
            ph_mask = None
        else:
            p_mask_aug_1 = tf.reduce_any(tf.cast(tf.tile(tf.expand_dims(p_mask, 2), [1, 1, PL, 1]), tf.bool), axis=3)
            p_mask_aug_2 = tf.reduce_any(tf.cast(tf.tile(tf.expand_dims(p_mask, 1), [1, PL, 1, 1]), tf.bool), axis=3)
            self_mask = p_mask_aug_1 & p_mask_aug_2

        h_logits = get_logits([p_aug_1, p_aug_2], None, True, wd=config.wd, mask=self_mask,
                              is_train=is_train, func=config.self_att_logit_func, scope='h_logits')  # [N, PL, HL]
        self_att = softsel(p_aug_2, h_logits)

        return self_att


def self_attention_layer(config, is_train, p, p_mask=None, scope=None):
    with tf.variable_scope(scope or "self_attention_layer"):
        self_att = self_attention(config, is_train, p, p_mask=p_mask)
        p0 = fuse_gate(config, is_train, p, self_att, scope="self_att_fuse_gate")
        return p0


def bi_attention(config, is_train, p, h, p_mask=None, h_mask=None, scope=None, h_value=None):  # [N, L, 2d]
    with tf.variable_scope(scope or "bi_attention"):
        PL = tf.shape(p)[1]
        HL = tf.shape(h)[1]
        p_aug = tf.tile(tf.expand_dims(p, 2), [1, 1, HL, 1])
        h_aug = tf.tile(tf.expand_dims(h, 1), [1, PL, 1, 1])  # [N, PL, HL, 2d]

        if p_mask is None:
            ph_mask = None
        else:
            p_mask_aug = tf.reduce_any(tf.cast(tf.tile(tf.expand_dims(p_mask, 2), [1, 1, HL, 1]), tf.bool), axis=3)
            h_mask_aug = tf.reduce_any(tf.cast(tf.tile(tf.expand_dims(h_mask, 1), [1, PL, 1, 1]), tf.bool), axis=3)
            ph_mask = p_mask_aug & h_mask_aug

        h_logits = get_logits([p_aug, h_aug], None, True, wd=config.wd, mask=ph_mask,
                              is_train=is_train, func="mul_linear", scope='h_logits')  # [N, PL, HL]
        h_a = softsel(h_aug, h_logits)
        p_a = softsel(p, tf.reduce_max(h_logits, 2))  # [N, 2d]
        p_a = tf.tile(tf.expand_dims(p_a, 1), [1, PL, 1])  #

        return h_a, p_a


def dense_net(config, denseAttention, is_train):
    with tf.variable_scope("dense_net"):
        dim = denseAttention.get_shape().as_list()[-1]
        act = tf.nn.relu if config.first_scale_down_layer_relu else None
        fm = tf.contrib.layers.convolution2d(denseAttention, int(dim * config.dense_net_first_scale_down_ratio),
                                             config.first_scale_down_kernel, padding="SAME", activation_fn=act)

        fm = dense_net_block(config, fm, config.dense_net_growth_rate, config.dense_net_layers,
                             config.dense_net_kernel_size, is_train, scope="first_dense_net_block")
        fm = dense_net_transition_layer(config, fm, config.dense_net_transition_rate, scope='second_transition_layer')
        fm = dense_net_block(config, fm, config.dense_net_growth_rate, config.dense_net_layers,
                             config.dense_net_kernel_size, is_train, scope="second_dense_net_block")
        fm = dense_net_transition_layer(config, fm, config.dense_net_transition_rate, scope='third_transition_layer')
        fm = dense_net_block(config, fm, config.dense_net_growth_rate, config.dense_net_layers,
                             config.dense_net_kernel_size, is_train, scope="third_dense_net_block")

        fm = dense_net_transition_layer(config, fm, config.dense_net_transition_rate, scope='fourth_transition_layer')

        shape_list = fm.get_shape().as_list()
        print(shape_list)
        out_final = tf.reshape(fm, [-1, shape_list[1] * shape_list[2] * shape_list[3]])
        return out_final


def dense_net_block(config, feature_map, growth_rate, layers, kernel_size, is_train, padding="SAME", act=tf.nn.relu,
                    scope=None):
    with tf.variable_scope(scope or "dense_net_block"):
        conv2d = tf.contrib.layers.convolution2d
        dim = feature_map.get_shape().as_list()[-1]

        list_of_features = [feature_map]
        features = feature_map
        for i in range(layers):
            ft = conv2d(features, growth_rate, (kernel_size, kernel_size), padding=padding, activation_fn=act)
            list_of_features.append(ft)
            features = tf.concat(list_of_features, axis=3)

        print("dense net block out shape")
        print(features.get_shape().as_list())
        return features


def dense_net_transition_layer(config, feature_map, transition_rate, scope=None):
    with tf.variable_scope(scope or "transition_layer"):
        out_dim = int(feature_map.get_shape().as_list()[-1] * transition_rate)
        feature_map = tf.contrib.layers.convolution2d(feature_map, out_dim, 1, padding="SAME", activation_fn=None)

        feature_map = tf.nn.max_pool(feature_map, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")

        print("Transition Layer out shape")
        print(feature_map.get_shape().as_list())
        return feature_map



def conv1d_block_old(input_reps, filter_sizes, shortcut, num_filters, layer_index, is_training=True, dropout_rate=0.2, use_cudnn=False, conv_type=None):
    for i in range(1):
        # filter_shape = [filter_sizes, inputs.get_shape()[2].value, num_filters]
        # W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        # inputs = tf.nn.conv1d(inputs, W, stride=1, padding="SAME", use_cudnn_on_gpu=True)
        # # inputs = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=filter_sizes, dilation_rate=2**layer_index, padding='causal')(inputs)
        # inputs = tf.layers.batch_normalization(inputs=inputs, momentum=0.99, epsilon=1e-3,
        #                                        center=True, scale=True, training=is_training)
        # inputs = tf.nn.relu(inputs)
        input_reps = tf.contrib.layers.batch_norm(input_reps, scale=True, is_training=is_training, updates_collections=None)
        input_reps = tf.nn.relu(input_reps)
        input_reps = tf.layers.conv1d(inputs=input_reps, filters=num_filters, kernel_size=filter_sizes, strides=1, dilation_rate=1,
                                      padding='same', activation=None)
        input_reps = dropout_layer(input_reps, dropout_rate, is_training=is_training)
    if conv_type == 'residual':
        return input_reps + shortcut
    elif conv_type == 'dense':
        return tf.concat(axis=2, values=[input_reps, shortcut])
    return input_reps


def my_cnn_layer_old(input_reps, input_length, input_dim, filter_sizes, num_filters, input_lengths=None, scope_name=None,
                 reuse=False, is_training=True,
                 dropout_rate=0.2, use_cudnn=False, conv_type=None):
    with tf.variable_scope(scope_name, reuse=reuse):
        input_reps = dropout_layer(input_reps, dropout_rate, is_training=is_training)
        # with tf.variable_scope("conv-0", reuse=reuse):
        # filter_shape = [1, input_dim, num_filters]
        # W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        # input_reps = tf.nn.conv1d(input_reps, W, stride=1, padding="SAME", use_cudnn_on_gpu=True)
        # input_reps = tf.layers.batch_normalization(inputs=input_reps, momentum=0.99, epsilon=1e-3,
        #                                        center=True, scale=True, training=is_training)
        # input_reps = tf.nn.relu(input_reps)
        input_reps = tf.contrib.layers.batch_norm(input_reps, scale=True, is_training=is_training, updates_collections=None)
        input_reps = tf.nn.relu(input_reps)
        input_reps = tf.layers.conv1d(inputs=input_reps, filters=num_filters, kernel_size=1, strides=1, dilation_rate=1,
                                      padding='same', activation=None)
        input_reps = dropout_layer(input_reps, dropout_rate, is_training=is_training)

        layers = [input_reps]
        for i, filter_size in enumerate(filter_sizes):
            # with tf.variable_scope("conv-%s" % (i + 1), reuse=reuse):
            if conv_type == 'residual':
                shortcut = layers[-1]
            elif conv_type == 'dense':
                shortcut = layers[-1] #tf.concat(axis=2, values=layers)
            else:
                shortcut = None
            conv_block = conv1d_block_old(input_reps=layers[-1], shortcut=shortcut, filter_sizes=filter_size,
                                      num_filters=num_filters, layer_index=i,
                                      is_training=is_training, dropout_rate=dropout_rate, use_cudnn=use_cudnn,
                                      conv_type=conv_type)
            layers.append(conv_block)

        multiscale_features = tf.concat(axis=1, values=layers)  # [batch_size, feature_len, feature_dim]
        # multiscale_features = tf.concat(axis=2, values=[tf.expand_dims(tf.reduce_mean(multiscale_features, axis=2), axis=-1),
        #                                                 tf.expand_dims(tf.reduce_max(multiscale_features, axis=2), axis=-1)])
        return multiscale_features


def conv1d_block(input_reps, filter_sizes, num_filters, layer_index, is_training=True, dropout_rate=0.2, use_cudnn=False, conv_type=None):
    for i in range(1):
        # filter_shape = [filter_sizes, inputs.get_shape()[2].value, num_filters]
        # W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        # inputs = tf.nn.conv1d(inputs, W, stride=1, padding="SAME", use_cudnn_on_gpu=True)
        # # inputs = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=filter_sizes, dilation_rate=2**layer_index, padding='causal')(inputs)
        # inputs = tf.layers.batch_normalization(inputs=inputs, momentum=0.99, epsilon=1e-3,
        #                                        center=True, scale=True, training=is_training)
        # inputs = tf.nn.relu(inputs)
        padding_type = 'same'
        if conv_type == 'temporal':
            padding_type = 'valid'
            padding = (filter_sizes - 1) * (2**i)
            input_reps = tf.pad(input_reps, tf.constant([(0, 0,), (1, 0), (0, 0)]) * padding)
            print(input_reps.get_shape())
        input_reps = tf.layers.conv1d(inputs=input_reps, filters=num_filters, kernel_size=filter_sizes, strides=1, dilation_rate=1,
                                      padding=padding_type, activation=None)
        input_reps = tf.nn.relu(input_reps)
        # input_reps = tf.contrib.layers.batch_norm(input_reps, scale=True, is_training=is_training, updates_collections=None)
        # input_reps = dropout_layer(input_reps, dropout_rate, is_training=is_training)
    return input_reps


def my_cnn_layer(input_reps, input_length, input_dim, filter_sizes, num_filters, input_lengths=None, scope_name=None,
                 reuse=False, is_training=True, dropout_rate=0.2, use_cudnn=False, conv_type=None):
    with tf.variable_scope(scope_name, reuse=reuse):
        # input_reps = dropout_layer(input_reps, dropout_rate, is_training=is_training)
        # with tf.variable_scope("conv-0", reuse=reuse):
        # filter_shape = [1, input_dim, num_filters]
        # W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        # input_reps = tf.nn.conv1d(input_reps, W, stride=1, padding="SAME", use_cudnn_on_gpu=True)
        # input_reps = tf.layers.batch_normalization(inputs=input_reps, momentum=0.99, epsilon=1e-3,
        #                                        center=True, scale=True, training=is_training)
        # input_reps = tf.nn.relu(input_reps)
        padding_type = 'same'
        # if conv_type == 'temporal':
        #     padding_type = 'valid'
        #     padding = (1 - 1) * 1
        #     input_reps = tf.pad(input_reps, tf.constant([(0, 0,), (1, 0), (0, 0)]) * padding)
        input_reps = tf.layers.conv1d(inputs=input_reps, filters=num_filters, kernel_size=1, strides=1, dilation_rate=1,
                                      padding=padding_type, activation=None)
        input_reps = tf.nn.relu(input_reps)
        # input_reps = tf.contrib.layers.batch_norm(input_reps, scale=True, is_training=is_training, updates_collections=None)
        # input_reps = dropout_layer(input_reps, dropout_rate, is_training=is_training)

        features = [input_reps]
        # for i, filter_size in enumerate(filter_sizes):
        #     # with tf.variable_scope("conv-%s" % (i + 1), reuse=reuse):
        #     shortcut = input_reps
        #     input_reps = conv1d_block(input_reps=input_reps, filter_sizes=filter_size,
        #                               num_filters=num_filters, layer_index=i + 1,
        #                               is_training=is_training, dropout_rate=dropout_rate, use_cudnn=use_cudnn,
        #                               conv_type=conv_type)
        #     features.append(input_reps)
        #     if conv_type == 'residual' or conv_type == 'temporal':
        #         input_reps = input_reps + shortcut
        #     elif conv_type == 'dense':
        #         input_reps = tf.concat(axis=2, values=[input_reps, shortcut])

        input_reps = tf.layers.conv1d(inputs=input_reps, filters=num_filters, kernel_size=2, strides=1, dilation_rate=1,
                                      padding=padding_type, activation=None)
        input_reps = tf.nn.relu(input_reps)
        features.append(input_reps)

        input_reps = tf.layers.conv1d(inputs=input_reps, filters=num_filters, kernel_size=2, strides=1, dilation_rate=1,
                                      padding=padding_type, activation=None)
        input_reps = tf.nn.relu(input_reps)
        features.append(input_reps)

        multiscale_features = tf.concat(axis=1, values=features)  # [batch_size, feature_len, feature_dim]
        return multiscale_features


def soft_attention_alignment(x1, x2):
    "Align text representation with neural soft attention"
    # x1: [b, s1, d]
    # x2: [b, s2, d]
    # att: [b, s1, s2]
    att = tf.einsum("abd,acd->abc", x1, x2)
    w_att_1 = tf.nn.softmax(att, dim=1)
    w_att_2 = tf.nn.softmax(att, dim=2)
    x1_att = tf.einsum("abd,acb->acd", x2, w_att_2)
    x2_att = tf.einsum("abd,abc->acd", x1, w_att_1)
    return x1_att, x2_att


def ms_feature_attention(input_reps, scope_name=None,
                 reuse=False, is_training=True,
                 dropout_rate=0.2, use_cudnn=False):
    with tf.variable_scope(scope_name, reuse=reuse):
        # pooling over each column/feature
        avg_ensem = tf.reduce_mean(input_reps, axis=2)
        max_ensem = tf.reduce_max(input_reps, axis=2)
        ensem_repres = tf.concat(axis=1, values=[avg_ensem, max_ensem])
        print(ensem_repres.get_shape())

        # mlp and softmax
        # att = tf.layers.dense(ensem_repres, ensem_repres.get_shape()[1].value, activation=tf.nn.relu)
        att = tf.layers.dense(ensem_repres, input_reps.get_shape()[1].value, activation=tf.nn.softmax)
        print(att.get_shape())
        output_reps = tf.einsum("abd,ab->ad", input_reps, att)
        return output_reps