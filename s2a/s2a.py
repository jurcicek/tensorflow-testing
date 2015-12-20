#!/usr/bin/env python3

import numpy as np
import sys

import tensorflow as tf

from tensorflow.python.ops.rnn_cell import LSTMCell

sys.path.extend(['..'])

import dataset

from tf_ext.bricks import linear, embedding, rnn, dense_to_one_hot

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_epochs', 1000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('decay', 0.9, 'Learning rate decay.')

"""
This code shows how to build and train a sequence to answer translation model.
"""


def train(train_set, test_set, idx2word, word2idx):
    embedding_size = 5
    vocabulary_length = len(idx2word)
    sequence_size = train_set['features'].shape[1]
    lstm_size = 5

    # inference model
    with tf.name_scope('model'):
        i = tf.placeholder("int32", name='input')
        o = tf.placeholder("int32", name='true_output')

        with tf.variable_scope("batch_size"):
            batch_size = tf.shape(i)[0]

        e = embedding(
                input=i,
                length=vocabulary_length,
                size=embedding_size,
                name='embedding'
        )

        with tf.name_scope("RNNCell"):
            cell = LSTMCell(lstm_size, input_size=embedding_size)
            state = cell.zero_state(batch_size, tf.float32)

        outputs, states = rnn(
                cell=cell,
                input=e,
                state=state,
                sequence_size=sequence_size,
                name='RNN'
        )

        final_state = states[-1]

        l = linear(
                input=final_state,
                input_size=cell.state_size,
                output_size=vocabulary_length,
                name='linear'
        )

        p_o_i = tf.nn.softmax(l, name="softmax_output")

    with tf.name_scope('loss'):
        one_hot_labels = dense_to_one_hot(o, vocabulary_length)
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
        train_op = tf.train.AdamOptimizer(FLAGS.learning_rate, name='trainer').minimize(loss)
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
        # print('Predictions')
        p_o_i = sess.run(p_o_i, feed_dict={i: test_set['features'], o: test_set['targets']})
        # print(p_o_i)
        print('Argmax predictions')
        print(np.argmax(p_o_i, 1).reshape((-1, 1)))


def main(_):
    train_set, test_set, idx2word, word2idx = dataset.dataset()

    train(train_set, test_set, idx2word, word2idx)


if __name__ == '__main__':
    tf.app.run()
