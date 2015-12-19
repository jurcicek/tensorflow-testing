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
        embedding = tf.Variable(
                tf.truncated_normal([length, size], stddev=3.0 / math.sqrt(float(length * size))),
                name='embedding'
        )

        y = tf.nn.embedding_lookup(embedding, input)

        y.length = length
        y.size = size
        y.embedding = embedding

    return y


def softmax_2d(input, n_classifiers, n_classes, name='softmax_2d'):
    with tf.name_scope(name):
        input = tf.reshape(input, [-1, n_classifiers, n_classes])
        e_x = tf.exp(input - tf.reduce_max(input, reduction_indices=2, keep_dims=True))
        # e_x[e_x < 1e-10] = 1e-10
        p_o_i = e_x / tf.reduce_sum(e_x, reduction_indices=2, keep_dims=True)

        return p_o_i


def dense_to_one_hot_2d(labels, n_classes):
    """ Converts dense representation of labels (e.g. 0, 1, 1) into one hot encoding ( e.g. (1, 0, 0), (0, 1, 0), etc.

    :param labels: integer input labels for a given batch
    :param n_classes: number of all different labels (classes)
    :return: a 2D tensor with encoded labels using one hot encoding
    """
    with tf.name_scope('mutlitarget_dense_to_one_hot'):
        batch_size = tf.shape(labels)[0]
        n_classifiers = tf.shape(labels)[1]
        labels = tf.reshape(labels, [-1, 1])
        indices_1 = tf.reshape(tf.tile(tf.expand_dims(tf.range(0, batch_size), 1), tf.pack([1, n_classifiers])),
                               [-1, 1])
        indices_2 = tf.tile(tf.expand_dims(tf.range(0, n_classifiers), 1), tf.pack([batch_size, 1]))
        concated = tf.concat(1, [indices_1, indices_2, labels])
        one_hot_labels = tf.sparse_to_dense(
                concated,
                tf.pack([batch_size, n_classifiers, n_classes]),
                1.0,
                0.0
        )

    return one_hot_labels


def dense_to_one_hot(labels, n_classes):
    """ Converts dense representation of labels (e.g. 0, 1, 1) into one hot encoding ( e.g. (1, 0, 0), (0, 1, 0), etc.

    :param labels: integer input labels for a given batch
    :param n_classes: number of all different labels (classes)
    :return: a 2D tensor with encoded labels using one hot encoding
    """

    batch_size = tf.size(labels)
    indices = tf.expand_dims(tf.range(0, batch_size), 1)
    concated = tf.concat(1, [indices, labels])
    one_hot_labels = tf.sparse_to_dense(
            concated,
            tf.pack([batch_size, n_classes]),
            1.0,
            0.0
    )

    return one_hot_labels
