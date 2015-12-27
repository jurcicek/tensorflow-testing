#!/usr/bin/env python3

import math
import tensorflow as tf


def linear(input, input_size, output_size, name='linear'):
    """Creates a linear transformation between two layers in a neural network.

    :param input: input into the linear block
    :param input_size: input dimension
    :param output_size: output dimension
    :param name: name of the operation
    """
    with tf.variable_scope(name):
        W = tf.get_variable(
                name='W',
                shape=[input_size, output_size],
                initializer=tf.truncated_normal_initializer(stddev=3.0 / math.sqrt(float(input_size * output_size))),
        )
        b = tf.get_variable(
                name='B',
                shape=[output_size],
                initializer=tf.truncated_normal_initializer(stddev=1e-9 / math.sqrt(float(input_size * output_size))),
        )

        y = tf.matmul(input, W) + b

        y.input_size = input_size
        y.output_size = output_size
        y.W = W
        y.b = b

    return y


def embedding(input, length, size, name='embedding'):
    """Embedding transformation between discrete input and continuous vector representation.

    :param input: input 2-D tensor (e.g. rows are examples, columns are words)
    :param length: int, the lengths of the table representing the embeddings. This is equal to the size of the all discrete
        inputs.
    :param size: int, size of vector representing the discrete inputs.
    :param name: str, name of the operation
    """
    with tf.name_scope(name):
        embedding_table = tf.get_variable(
                name='embedding_table',
                shape=[length, size],
                initializer=tf.truncated_normal_initializer(stddev=3.0 / math.sqrt(float(length * size))),
        )

        y = tf.nn.embedding_lookup(embedding_table, input)

        y.length = length
        y.size = size
        y.embedding_table = embedding_table

    return y


def softmax_2d(input, n_classifiers, n_classes, name='softmax_2d'):
    with tf.name_scope(name):
        input = tf.reshape(input, [-1, n_classifiers, n_classes])
        e_x = tf.exp(input - tf.reduce_max(input, reduction_indices=2, keep_dims=True)) + 1e-10
        p_o_i = e_x / tf.reduce_sum(e_x, reduction_indices=2, keep_dims=True)

        return p_o_i


def rnn(cell, inputs, initial_state, name='RNN', reuse=False):
    """Forward recurrent neural network.

    :param cell: An instance of RNNCell
    :param inputs: A list of tensors, each a tensor of shape [batch_size, cell.input_size]
    :param initial_state: A tensors, each a tensor of shape [batch_size, cell.state_size]
    :param name: A name of the variable scope for created or reused variables
    :param reuse: True if any created variables should be reused otherwise None.
    :return:
    """
    with tf.variable_scope(name, reuse=reuse):
        outputs = []
        states = [initial_state]

        for j, input in enumerate(inputs):
            with tf.variable_scope(tf.get_variable_scope(), reuse=True if j > 0 else None):
                output, state = cell(input, states[-1])

                outputs.append(output)
                states.append(state)

    # remove the initial state
    states = states[1:]

    return outputs, states


def brnn(cell_fw, cell_bw, inputs, initial_state_fw, initial_state_bw=None, name='BidirectionalRNN', reuse=False):
    """Bidirectional recurrent neural network.

    :param cell_fw: An instance of RNNCell, for the forward pass
    :param cell_bw: An instance of RNNCell, for the backward pass
    :param inputs: A list of tensors, each a tensor of shape [batch_size, cell.input_size]
    :param initial_state_fw: A tensors, each a tensor of shape [batch_size, cell.state_size]
    :param initial_state_bw: (Optional) A tensors, each a tensor of shape [batch_size, cell.state_size]
    :param name: A name of the variable scope for created or reused variables
    :param reuse: True if any created variables should be reused otherwise None.
    :return:
    """
    if not initial_state_bw:
        initial_state_bw = initial_state_fw

    with tf.variable_scope(name, reuse=reuse):
        outputs_fw, states_fw = rnn(cell_fw, inputs, initial_state_fw, name='ForwardRNN', reuse=reuse)
        outputs_bw, states_bw = rnn(cell_bw, reversed(inputs), initial_state_bw, name='BackwardRNN', reuse=reuse)

        # print(outputs_fw, states_fw)
        # print(outputs_bw, states_bw)

        outputs = [tf.concat(1, [fw, bw]) for fw, bw in zip(outputs_fw, reversed(outputs_bw))]
        states = [tf.concat(1, [fw, bw]) for fw, bw in zip(states_fw, reversed(states_bw))]

    return outputs, states


def rnn_decoder(cell, inputs, initial_state, embedding_size, embedding_length, sequence_length,
                name='RNNDecoder', reuse=False, use_inputs_prob=0.0):
    with tf.variable_scope(name, reuse=reuse):
        # print(tf.get_variable_scope().reuse, tf.get_variable_scope().name)
        with tf.name_scope("embedding"):
            batch_size = tf.shape(initial_state)[0]
            embedding_table = tf.get_variable(
                    name='embedding_table',
                    shape=[embedding_length, embedding_size],
                    initializer=tf.truncated_normal_initializer(
                            stddev=3.0 / math.sqrt(float(embedding_length * embedding_size))
                    ),
            )
            # 0 is index for _SOS_ (start of sentence symbol)
            initial_embedding = tf.nn.embedding_lookup(embedding_table, tf.zeros(tf.pack([batch_size]), tf.int32))

        states = [initial_state]
        outputs = []
        outputs_softmax = []
        decoder_outputs_argmax_embedding = []

        for j in range(sequence_length):
            with tf.variable_scope(tf.get_variable_scope(), reuse=True if j > 0 else None):
                # get input :
                #   either feedback the previous decoder argmax output
                #   or use the provided input (note that you have to use the previous input (index si therefore -1)
                input = initial_embedding
                if j > 0:
                    true_input = tf.nn.embedding_lookup(embedding_table, inputs[j-1])
                    decoded_input = decoder_outputs_argmax_embedding[-1]
                    choice = tf.floor(tf.random_uniform([1], use_inputs_prob, 1+use_inputs_prob, tf.float32))
                    input = choice * true_input + (1.0 - choice) * decoded_input

                # print(tf.get_variable_scope().reuse, tf.get_variable_scope().name)
                output, state = cell(input, states[-1])

                projection = linear(
                        input=output,
                        input_size=cell.output_size,
                        output_size=embedding_length,
                        name='output_linear_projection'
                )

                outputs.append(projection)
                states.append(state)

                softmax = tf.nn.softmax(projection, name="output_softmax")
                # we do no compute the gradient trough argmax
                output_argmax = tf.stop_gradient(tf.argmax(softmax, 1))
                # we do no compute the gradient for embeddings when used with noisy argmax outputs
                output_argmax_embedding = tf.stop_gradient(tf.nn.embedding_lookup(embedding_table, output_argmax))
                decoder_outputs_argmax_embedding.append(output_argmax_embedding)

                outputs_softmax.append(tf.expand_dims(softmax, 1))

    # remove the initial state
    states = states[1:]

    return states, outputs, outputs_softmax


def dense_to_one_hot(labels, n_classes):
    with tf.name_scope('dense_to_one_hot'):
        indices = tf.where(tf.greater_equal(labels, tf.zeros_like(labels)))
        concated = tf.concat(1, [tf.to_int32(indices), tf.reshape(labels, [-1, 1])])
        dim = tf.concat(0, [tf.shape(labels), tf.reshape(n_classes, [-1])])

        one_hot_labels = tf.sparse_to_dense(
                concated,
                dim,
                1.0,
                0.0
        )

        return one_hot_labels
