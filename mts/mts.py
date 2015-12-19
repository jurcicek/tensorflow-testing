#!/usr/bin/env python3

import math
import numpy as np
import h5py

import tensorflow as tf
from numpy.core.numeric import indices

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 1.0, 'Initial learning rate.')
flags.DEFINE_float('decay', 0.9, 'Learning rate decay.')

"""
This code shows how to train softmax outputs for multiple targets.

Imagine a NN rather complex which serves as set of binary classifiers. You do not want to build N independent
binary classifiers as the hidden layers are rather big and computationally expensive.

In this case we train a NN for training 3 binary classifiers evaluating the OR, AND, and XOR logical operations.

"""


def prepare_dataset():
    train_features = np.array(
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
            ],
            dtype=np.float32
    )
    test_features = np.array(
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
            ],
            dtype=np.float32
    )
    train_targets = np.array(
            [
                [0, 0, 0],
                [1, 0, 1],
                [1, 0, 1],
                [1, 1, 0],
                [0, 0, 0],
                [1, 0, 1],
                [1, 0, 1],
                [1, 1, 0],
            ],
            dtype=np.uint8
    )
    test_targets = np.array(
            [
                [0, 0, 0],
                [1, 0, 1],
                [1, 0, 1],
                [1, 1, 0],
            ],
            dtype=np.uint8
    )

    # f = h5py.File('data/dataset.hdf5', mode='w')
    #
    # features = f.create_dataset(
    #         'features',
    #         (
    #             train_features.shape[0] + test_features.shape[0],
    #             train_features.shape[1],
    #         ),
    #         dtype='float32'
    #
    # )
    #
    # features.dims[0].label = 'example'
    # features.dims[1].label = 'feature'
    #
    # features[...] = np.vstack(
    #         [train_features, test_features]
    # )
    #
    # targets = f.create_dataset(
    #         'targets',
    #         (
    #             train_targets.shape[0] + test_targets.shape[0],
    #             train_targets.shape[1],
    #         ),
    #         dtype='uint8'
    # )
    # targets.dims[0].label = 'example'
    # targets.dims[1].label = 'indexs'
    #
    # targets[...] = np.vstack(
    #         [train_targets, test_targets]
    # )
    #
    # f.flush()
    # f.close()

    train_set = {
        'features': train_features,
        'targets':  train_targets
    }
    test_set = {
        'features': test_features,
        'targets':  test_targets
    }

    return train_set, test_set


def Linear(input_dim, output_dim, input, name='Linear'):
    """Initialise a linear transformation between two layers in a neural network.

    :param input: input into the linear block
    :param input_dim: input dimension
    :param output_dim: output dimension
    :param name: name of the operation
    """
    with tf.name_scope(name):
        W = tf.Variable(
                tf.random_normal([input_dim, output_dim], stddev=3.0 / math.sqrt(float(input_dim * output_dim))),
                name='W'
        )
        b = tf.Variable(
                tf.zeros([output_dim]),
                name='b'
        )

        y = tf.matmul(input, W) + b

        y.input_dim = input_dim
        y.output_dim = output_dim
        y.W = W
        y.b = b

    return y


def mutlitarget_dense_to_one_hot(labels, n_classes):
    """ Converts dense representation of labels (e.g. 0, 1, 1) into one hot encoding ( e.g. (1, 0, 0), (0, 1, 0), etc.

    :param labels: integer input labels for a given batch
    :param n_classes: number of all different labels (classes)
    :return: a 2D tensor with encoded labels using one hot encoding
    """
    with tf.name_scope('mutlitarget_dense_to_one_hot'):
        batch_size = tf.shape(labels)[0]
        n_classifiers = tf.shape(labels)[1]
        labels = tf.reshape(labels, [-1, 1])
        indices_1 = tf.reshape(tf.tile(tf.expand_dims(tf.range(0, batch_size), 1), tf.pack([1, n_classifiers])), [-1, 1])
        indices_2 = tf.tile(tf.expand_dims(tf.range(0, n_classifiers), 1), tf.pack([batch_size, 1]))
        concated = tf.concat(1, [indices_1, indices_2, labels])
        onehot_labels = tf.sparse_to_dense(
                concated,
                tf.pack([batch_size, n_classifiers, n_classes]),
                1.0,
                0.0
        )

    return onehot_labels



def train(train_set, test_set):
    input_dim = 2

    n_classes = 2
    n_classifiers = 3

    # inference model
    with tf.name_scope('model'):
        i = tf.placeholder("float", name='input')
        o = tf.placeholder("int32", name='true_output')

        l1 = Linear(
                input_dim=input_dim,
                output_dim=3,
                input=i,
                name='linear_1'
        )

        a1 = tf.nn.tanh(l1, name='tanh_activation')

        l2 = Linear(
                input_dim=l1.output_dim,
                output_dim=n_classes * n_classifiers,
                input=a1,
                name='linear_2'
        )

        with tf.name_scope('multitarget_softmax_activation'):
            l2 = tf.reshape(l2, [-1, n_classifiers, n_classes])
            e_x = tf.exp(l2 - tf.reduce_max(l2, reduction_indices=2, keep_dims=True))
            p_o_i = e_x / tf.reduce_sum(e_x, reduction_indices=2, keep_dims=True)

    with tf.name_scope('loss'):
        one_hot_labels = mutlitarget_dense_to_one_hot(o, n_classes)
        loss = tf.reduce_mean(-one_hot_labels * tf.log(p_o_i), name='loss')
        tf.scalar_summary('loss', loss)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(one_hot_labels, 2), tf.argmax(p_o_i, 2))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        tf.scalar_summary('accuracy', accuracy)

    with tf.Session() as sess:
        # Merge all the summaries and write them out to ./log
        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter('./log', sess.graph_def)

        # training
        train_op = tf.train.RMSPropOptimizer(FLAGS.learning_rate, FLAGS.decay, name='trainer').minimize(loss)
        tf.initialize_all_variables().run()

        for epoch in range(FLAGS.max_steps):
            sess.run(train_op, feed_dict={i: train_set['features'], o: train_set['targets']})

            if epoch % max(int(FLAGS.max_steps / 100), 1) == 0:
                summary, lss, acc = sess.run([merged, loss, accuracy],
                                             feed_dict={i: test_set['features'], o: test_set['targets']})
                writer.add_summary(summary, epoch)
                print()
                print('Epoch: {epoch}'.format(epoch=epoch))
                print(' - accuracy = {acc}'.format(acc=acc))
                print(' - loss     = {lss}'.format(lss=lss))

        print('Test features')
        print(test_set['features'])
        print('Test targets')
        print(test_set['targets'])
        print('Predictions')
        p_o_i = sess.run(p_o_i, feed_dict={i: test_set['features'], o: test_set['targets']})
        print(p_o_i)
        print('Argmax predictions')
        print(np.matrix(np.argmax(p_o_i, 2)))


def main(_):
    train_set, test_set = prepare_dataset()
    train(train_set, test_set)


if __name__ == '__main__':
    tf.app.run()
