#!/usr/bin/env python3

import numpy as np
import sys

import tensorflow as tf

from tensorflow.python.ops.rnn_cell import LSTMCell

sys.path.extend(['..'])

import dataset

from tf_ext.bricks import linear, embedding, rnn, dense_to_one_hot, rnn_decoder

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_epochs', 1000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('decay', 0.9, 'Learning rate decay.')

"""
This code shows how to build and train a sequence to answer translation model.
"""


def train(train_set, test_set, idx2word, word2idx):
    encoder_lstm_size = 5
    encoder_embedding_size = 5
    encoder_vocabulary_length = len(idx2word)
    encoder_sequence_length = train_set['features'].shape[1]

    decoder_lstm_size = 5
    decoder_embedding_size = 5
    decoder_vocabulary_length = len(idx2word)
    decoder_sequence_length = train_set['targets'].shape[1]

    # inference model
    with tf.name_scope('model'):
        i = tf.placeholder("int32", name='input')
        o = tf.placeholder("int32", name='true_output')

        with tf.variable_scope("batch_size"):
            batch_size = tf.shape(i)[0]

        encoder_embedding = embedding(
                input=i,
                length=encoder_vocabulary_length,
                size=encoder_embedding_size,
                name='encoder_embedding'
        )

        with tf.name_scope("RNNEncoderCell"):
            cell = LSTMCell(
                    num_units=encoder_lstm_size,
                    input_size=encoder_embedding_size,
                    use_peepholes=False
            )
            initial_state = cell.zero_state(batch_size, tf.float32)

        encoder_outputs, encoder_states = rnn(
                cell=cell,
                inputs=[encoder_embedding[:, j, :] for j in range(encoder_sequence_length)],
                initial_state=initial_state,
                name='RNNForwardEncoder'
        )

        final_encoder_state = encoder_states[-1]

        with tf.name_scope("RNNDecoderCell"):
            cell = LSTMCell(
                    num_units=decoder_lstm_size,
                    input_size=decoder_embedding_size,
                    use_peepholes=False,
            )

        decoder_states, decoder_outputs, decoder_outputs_softmax = rnn_decoder(
                cell=cell,
                initial_state=final_encoder_state,
                embedding_size=decoder_embedding_size,
                embedding_length=decoder_vocabulary_length,
                sequence_length=decoder_sequence_length,
                name='RNNDecoder'
        )

        p_o_i = tf.concat(1, decoder_outputs_softmax)

    with tf.name_scope('loss'):
        one_hot_labels = dense_to_one_hot(o, decoder_vocabulary_length)
        loss = tf.reduce_mean(-one_hot_labels * tf.log(p_o_i), name='loss')
        # loss = tf.constant(0.0, dtype=tf.float32)
        tf.scalar_summary('loss', loss)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(one_hot_labels, 2), tf.argmax(p_o_i, 2))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        # accuracy = tf.constant(0.0, dtype=tf.float32)
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
        print('Shape of targets:', test_set['targets'].shape)
        print(test_set['targets'])
        print('Predictions')
        p_o_i = sess.run(p_o_i, feed_dict={i: test_set['features'], o: test_set['targets']})
        p_o_i_argmax = np.argmax(p_o_i, 2)
        print('Shape of predictions:', p_o_i.shape)
        print('Argmax predictions')
        print(p_o_i_argmax)
        print()
        for i in range(p_o_i_argmax.shape[0]):
            for j in range(p_o_i_argmax.shape[1]):
                w = idx2word[p_o_i_argmax[i, j]]
                if w not in ['_SOS_', '_EOS_']:
                    print(w, end=' ')
            print()


def main(_):
    train_set, test_set, idx2word, word2idx = dataset.dataset()

    train(train_set, test_set, idx2word, word2idx)


if __name__ == '__main__':
    tf.app.run()
