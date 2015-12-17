#!/usr/bin/env python3
import math
import numpy as np
import h5py

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('decay', 0.9, 'Learning rate decay.')


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
                [0],
                [1],
                [1],
                [0],
                [0],
                [1],
                [1],
                [0],
            ],
            dtype=np.uint8
    )

    test_targets = np.array(
            [
                [0],
                [1],
                [1],
                [0],
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
    # )
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
    #             1,  # train_targets.shape[1],
    #         ),
    #         dtype='uint8'
    # )
    # targets.dims[0].label = 'example'
    # targets.dims[1].label = 'label'
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


def Linear( input_dim, output_dim, input, name='Linear'):
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

    return y


def dense_to_one_hot(labels, n_clases):
    """ Converts dense representation of labels (e.g. 0, 1, 1) into one hot encoding ( e.g. (1, 0, 0), (0, 1, 0), etc.

    :param labels: integer input labels for a given batch
    :param n_clases: number of all different labels (classes)
    :return: a 2D tensor with encoded labels using one hot encoding
    """

    batch_size = tf.size(labels)
    indices = tf.expand_dims(tf.range(0, batch_size), 1)
    concated = tf.concat(1, [indices, labels])
    onehot_labels = tf.sparse_to_dense(
            concated,
            tf.pack([batch_size, n_clases]),
            1.0,
            0.0
    )

    return onehot_labels


def train(train_set, test_set):
    input_dim = 2
    n_classes = 2

    # inference model
    with tf.name_scope('model'):
        i = tf.placeholder("float", name='input')
        o = tf.placeholder("int32", name='true_output')

        l1 = Linear(input_dim, 3, input=i, name='linear_1')

        a1 = tf.nn.tanh(l1, name='tanh_activation')

        l2 = Linear(3, n_classes, input=a1, name='linear_2')

        p_o_i = tf.nn.softmax(l2, name="softmax_output")

    with tf.name_scope('loss'):
        onehot_labels = dense_to_one_hot(o, n_classes)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(l2, onehot_labels), name='loss')
        tf.scalar_summary('loss', loss)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(onehot_labels, 1), tf.argmax(p_o_i, 1))
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
                summary, lss, acc = sess.run([merged, loss, accuracy], feed_dict={i: test_set['features'], o: test_set['targets']})
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
        print(np.matrix(np.argmax(p_o_i, 1)).transpose())


def main(_):
    train_set, test_set = prepare_dataset()
    train(train_set, test_set)


if __name__ == '__main__':
    tf.app.run()
