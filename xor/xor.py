#!/usr/bin/env python3
import numpy as np
import sys

import tensorflow as tf

sys.path.extend(['..'])
from tf_ext.bricks import linear, dense_to_one_hot

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_epochs', 1000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 1.0, 'Initial learning rate.')
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


    train_set = {
        'features': train_features,
        'targets':  train_targets
    }
    test_set = {
        'features': test_features,
        'targets':  test_targets
    }

    return train_set, test_set


def train(train_set, test_set):
    input_dim = 2
    n_classes = 2

    # inference model
    with tf.name_scope('model'):
        i = tf.placeholder("float", name='input')
        o = tf.placeholder("int32", name='true_output')

        l1 = linear(
                input=i,
                input_size=input_dim,
                output_size=3,
                name='linear_1'
        )

        a1 = tf.nn.tanh(l1, name='tanh_activation')

        l2 = linear(
                input=a1,
                input_size=l1.input_size,
                output_size=n_classes,
                name='linear_2'
        )

        p_o_i = tf.nn.softmax(l2, name="softmax_output")

    with tf.name_scope('loss'):
        one_hot_labels = dense_to_one_hot(o, n_classes)
        loss = tf.reduce_mean(-one_hot_labels * tf.log(p_o_i), name='loss')
        tf.scalar_summary('loss', loss)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(one_hot_labels, 1), tf.argmax(p_o_i, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        tf.scalar_summary('accuracy', accuracy)

    with tf.Session() as sess:
        # Merge all the summaries and write them out to ./log
        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter('./log', sess.graph_def)
        saver = tf.train.Saver()

        # training
        train_op = tf.train.RMSPropOptimizer(FLAGS.learning_rate, FLAGS.decay, name='trainer').minimize(loss)
        tf.initialize_all_variables().run()

        for epoch in range(FLAGS.max_epochs):
            sess.run(train_op, feed_dict={i: train_set['features'], o: train_set['targets']})

            if epoch % max(int(FLAGS.max_epochs / 100), 1) == 0:
                summary, lss, acc = sess.run([merged, loss, accuracy],
                                             feed_dict={i: test_set['features'], o: test_set['targets']})
                writer.add_summary(summary, epoch)
                print()
                print('Epoch: {epoch}'.format(epoch=epoch))
                print(' - accuracy = {acc}'.format(acc=acc))
                print(' - loss     = {lss}'.format(lss=lss))

        save_path = saver.save(sess, "model.ckpt")
        print()
        print("Model saved in file: %s" % save_path)
        print()

        print('Test features')
        print(test_set['features'])
        print('Test targets')
        print(test_set['targets'])
        print('Predictions')
        p_o_i = sess.run(p_o_i, feed_dict={i: test_set['features'], o: test_set['targets']})
        print(p_o_i)
        print('Argmax predictions')
        print(np.argmax(p_o_i, 1).reshape((-1, 1)))


def main(_):
    train_set, test_set = prepare_dataset()
    train(train_set, test_set)


if __name__ == '__main__':
    tf.app.run()
