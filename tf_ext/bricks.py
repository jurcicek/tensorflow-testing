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
    with tf.name_scope(name):
        W = tf.Variable(
                tf.truncated_normal([input_size, output_size], stddev=3.0 / math.sqrt(float(input_size * output_size))),
                name='W'
        )
        b = tf.Variable(
                tf.zeros([output_size]),
                name='b'
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
        embedding_table = tf.Variable(
                tf.truncated_normal([length, size], stddev=3.0 / math.sqrt(float(length * size))),
                name='embedding'
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
    with tf.variable_scope(name):
        outputs = []
        states = [initial_state]

        for j in range(len(inputs)):
            with tf.variable_scope(tf.get_variable_scope(), reuse=True if j > 0 or reuse else None):
                output, state = cell(inputs[j], states[-1])

                outputs.append(outputs)
                states.append(state)

    return outputs, states


def rnn_decoder(cell, initial_state, embedding_size, embedding_length, sequence_length, name='RNNDecoder', reuse=False):
    with tf.variable_scope(name):
        with tf.name_scope("decoder_embedding"):
            with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
                batch_size = tf.shape(initial_state)[0]
                initial_embedding = tf.zeros(tf.pack([batch_size, embedding_size]), tf.float32)

                embedding_table = tf.Variable(
                        tf.truncated_normal(
                                [embedding_length, embedding_size],
                                stddev=3.0 / math.sqrt(float(embedding_length * embedding_size))
                        ),
                        name='decoder_embedding'
                )

        decoder_states = [initial_state]
        decoder_outputs = []
        decoder_outputs_softmax = []
        decoder_outputs_argmax_embedding = [initial_embedding]

        for j in range(sequence_length):
            with tf.variable_scope(tf.get_variable_scope(), reuse=True if j > 0 or reuse else None):
                output, state = cell(decoder_outputs_argmax_embedding[-1], decoder_states[-1])

                projection = linear(
                        input=output,
                        input_size=cell.output_size,
                        output_size=embedding_length,
                        name='decoder_output_linear_projection'
                )

                decoder_outputs.append(projection)
                decoder_states.append(state)

                softmax = tf.nn.softmax(projection, name="decoder_output_softmax")
                output_argmax = tf.argmax(softmax, 1)
                output_argmax_embedding = tf.nn.embedding_lookup(embedding_table, output_argmax)
                decoder_outputs_argmax_embedding.append(output_argmax_embedding)

                decoder_outputs_softmax.append(tf.expand_dims(softmax, 1))

    return decoder_states, decoder_outputs, decoder_outputs_softmax


def dense_to_one_hot(labels, n_classes):
    with tf.name_scope('dense_to_one_hot_x'):
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
