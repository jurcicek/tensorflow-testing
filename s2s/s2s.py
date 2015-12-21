#!/usr/bin/env python3
import math
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
    encoder_lstm_size = 5
    encoder_embedding_size = 5
    encoder_vocabulary_length = len(idx2word)
    encoder_sequence_size = train_set['features'].shape[1]

    decoder_lstm_size = 5
    decoder_embedding_size = 5
    decoder_vocabulary_length = len(idx2word)
    decoder_sequence_size = train_set['targets'].shape[1]

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
                name='embedding'
        )

        with tf.name_scope("RNNEncoderCell"):
            cell = LSTMCell(encoder_lstm_size, input_size=encoder_embedding_size)
            initial_state = cell.zero_state(batch_size, tf.float32)

        encoder_outputs, encoder_states = rnn(
                cell=cell,
                input=encoder_embedding,
                initial_state=initial_state,
                sequence_size=encoder_sequence_size,
                name='RNNForwardEncoder'
        )

        final_encoder_state = encoder_states[-1]

        with tf.name_scope("RNNDecoderCell"):
            cell = LSTMCell(encoder_lstm_size, input_size=decoder_embedding_size)

        initial_embedding = tf.zeros(tf.pack([batch_size, decoder_embedding_size]), tf.float32)

        decoder_embedding = tf.Variable(
                tf.truncated_normal(
                        [decoder_vocabulary_length, decoder_embedding_size],
                        stddev=3.0 / math.sqrt(float(decoder_vocabulary_length * decoder_embedding_size))
                ),
                name='decoder_embedding'
        )

        name = 'RNNDecoder'
        with tf.variable_scope(name):
            decoder_states = [final_encoder_state]
            decoder_outputs = []
            decoder_outputs_argmax = []
            decoder_outputs_argmax_embedding = [initial_embedding]

            for j in range(decoder_sequence_size):
                with tf.variable_scope(tf.get_variable_scope(), reuse=True if j > 0 else None):

                    output, state = cell(decoder_outputs_argmax_embedding[-1], decoder_states[-1])

                    decoder_outputs.append(output)
                    decoder_states.append(state)

                    output_argmax = tf.argmax(output, 1)
                    decoder_outputs_argmax.append(output_argmax)
                    output_argmax_embedding = tf.nn.embedding_lookup(decoder_embedding, output_argmax)
                    decoder_outputs_argmax_embedding.append(output_argmax_embedding)

        name = 'RNNDecoderSoftmax'
        with tf.variable_scope(name):
            linears = []
            softmaxs = []

            for j, output in enumerate(decoder_outputs):
                l = linear(
                        input=output,
                        input_size=decoder_lstm_size,
                        output_size=decoder_vocabulary_length,
                        name='linear'
                )

                linears.append(l)

                p_o_i = tf.nn.softmax(l, name="softmax_output")

                p_o_i = tf.expand_dims(p_o_i, 1)
                softmaxs.append(p_o_i)

            p_o_i = tf.concat(1, softmaxs)

    with tf.name_scope('loss'):
        # one_hot_labels = dense_to_one_hot(o, encoder_vocabulary_length)
        # loss = tf.reduce_mean(-one_hot_labels * tf.log(p_o_i), name='loss')
        loss = tf.constant(0.0, dtype=tf.float32)
        tf.scalar_summary('loss', loss)

    with tf.name_scope('accuracy'):
        # correct_prediction = tf.equal(tf.argmax(one_hot_labels, 1), tf.argmax(p_o_i, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        accuracy = tf.constant(0.0, dtype=tf.float32)
        tf.scalar_summary('accuracy', accuracy)

    with tf.Session() as sess:
        # Merge all the summaries and write them out to ./log
        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter('./log', sess.graph_def)
        saver = tf.train.Saver()

        # training
        # train_op = tf.train.AdamOptimizer(FLAGS.learning_rate, name='trainer').minimize(loss)
        tf.initialize_all_variables().run()

        for epoch in range(FLAGS.max_epochs):
            # sess.run(train_op, feed_dict={i: train_set['features'], o: train_set['targets']})

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
        print('Shape of predictions:', p_o_i.shape)
        print('Argmax predictions')
        print(np.argmax(p_o_i, 2))


def main(_):
    train_set, test_set, idx2word, word2idx = dataset.dataset()

    train(train_set, test_set, idx2word, word2idx)


if __name__ == '__main__':
    tf.app.run()
