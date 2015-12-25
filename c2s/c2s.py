#!/usr/bin/env python3
from random import shuffle

import numpy as np
import sys

import tensorflow as tf

from tensorflow.python.ops.rnn_cell import LSTMCell, GRUCell

sys.path.extend(['..'])

import dataset

from tf_ext.bricks import embedding, rnn, rnn_decoder, dense_to_one_hot

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_epochs', 1000, 'Number of epochs to run trainer.')
flags.DEFINE_integer('batch_size', 2, 'Number of training examples in a batch.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('beta1', 0.009, 'AdamOptimiser 1st moment decay.')
flags.DEFINE_float('beta2', 0.999, 'AdamOptimiser 2nd moment decay.')
flags.DEFINE_float('epsilon', 1e-8, 'AdamOptimiser epsilon.')
flags.DEFINE_float('regularization', 1e-2, 'Weight of regularization.')
flags.DEFINE_float('print_variables', False, 'Print all trainable variables.')

"""
This code shows how to build and train a sequence to answer translation model.
"""


def train(train_set, test_set, idx2word, word2idx):
    with tf.variable_scope("conversation_length"):
        conversation_length = train_set['features'].shape[1]

    encoder_lstm_size = 5
    encoder_embedding_size = 5
    encoder_vocabulary_length = len(idx2word)
    with tf.variable_scope("encoder_sequence_length"):
        encoder_sequence_length = train_set['features'].shape[2]

    decoder_lstm_size = 5
    decoder_embedding_size = 5
    decoder_vocabulary_length = len(idx2word)
    with tf.variable_scope("decoder_sequence_length"):
        decoder_sequence_length = train_set['targets'].shape[2]

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

        with tf.name_scope("RNNTurnEncoderCell"):
            cell = LSTMCell(
                    num_units=encoder_lstm_size,
                    input_size=encoder_embedding_size,
                    use_peepholes=True
            )
            initial_state = cell.zero_state(batch_size, tf.float32)

        # the input data has this dimensions
        # [#batch, #turn (sentence) in a conversation (a dialogue), #word in a turn (a sentence), # embedding dimension]

        # encode all turns along the word axis
        encoder_outputs_2d = []
        encoder_states_2d = []

        for turn in range(conversation_length):
            encoder_outputs, encoder_states = rnn(
                    cell=cell,
                    inputs=[encoder_embedding[:, turn, word, :] for word in range(encoder_sequence_length)],
                    initial_state=initial_state,
                    name='RNNTurnForwardEncoder',
                    reuse=True if turn > 0 else None
            )
            encoder_outputs_2d.append(encoder_outputs)

            # print(encoder_states[-1])
            encoder_states = tf.concat(1, tf.expand_dims(encoder_states[-1], 1))
            # print(encoder_states)
            encoder_states_2d.append(encoder_states)

        encoder_states_2d = tf.concat(1, encoder_states_2d)
        # print('encoder_states_2d', encoder_states_2d)

        # encode all conversations along the turn axis
        with tf.name_scope("RNNConversationEncoderCell"):
            cell = LSTMCell(
                    num_units=encoder_lstm_size,
                    input_size=cell.state_size,
                    use_peepholes=True
            )
            initial_state = cell.zero_state(batch_size, tf.float32)

        encoder_outputs, encoder_states = rnn(
                cell=cell,
                inputs=[encoder_states_2d[:, turn, :] for turn in range(conversation_length)],
                initial_state=initial_state,
                name='RNNConversationForwardEncoder',
                reuse=None
        )

        decoder_p_o_i_2d = []

        with tf.name_scope("RNNDecoderCell"):
            cell = LSTMCell(
                    num_units=decoder_lstm_size,
                    input_size=decoder_embedding_size,
                    use_peepholes=True,
            )

        # decode all conversations along the turn axis
        for turn in range(conversation_length):
            final_encoder_state = encoder_states[turn]

            decoder_states, decoder_outputs, decoder_outputs_softmax = rnn_decoder(
                    cell=cell,
                    initial_state=final_encoder_state,
                    embedding_size=decoder_embedding_size,
                    embedding_length=decoder_vocabulary_length,
                    sequence_length=decoder_sequence_length,
                    name='RNNDecoder',
                    reuse=True if turn > 0 else None
            )

            # print('decoder_outputs_softmax', decoder_outputs_softmax[0])
            p_o_i = tf.concat(1, decoder_outputs_softmax)
            p_o_i = tf.expand_dims(p_o_i, 1)
            # print('p_o_i', p_o_i)

            decoder_p_o_i_2d.append(p_o_i)

        p_o_i = tf.concat(1, decoder_p_o_i_2d)
        # print(p_o_i)

    if FLAGS.print_variables:
        for v in tf.trainable_variables():
            print(v.name)

    with tf.name_scope('loss'):
        one_hot_labels = dense_to_one_hot(o, decoder_vocabulary_length)
        loss = tf.reduce_mean(- one_hot_labels * tf.log(p_o_i), name='loss')
        for v in tf.trainable_variables():
            for n in ['/W', '/B']:
                if n in v.name:
                    print('Regularization using', v.name)
                    loss += FLAGS.regularization * tf.reduce_mean(tf.pow(v, 2))
        tf.scalar_summary('loss', loss)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(one_hot_labels, 3), tf.argmax(p_o_i, 3))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        # accuracy = tf.constant(0.0, dtype=tf.float32)
        tf.scalar_summary('accuracy', accuracy)

    with tf.Session() as sess:
        # Merge all the summaries and write them out to ./log
        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter('./log', sess.graph_def)
        saver = tf.train.Saver()

        # training
        train_op = tf.train.AdamOptimizer(
                FLAGS.learning_rate,
                beta1=FLAGS.beta1,
                beta2=FLAGS.beta2,
                epsilon=FLAGS.epsilon,
                name='trainer').minimize(loss)
        tf.initialize_all_variables().run()

        # prepare batch indexes
        train_set_size = train_set['features'].shape[0]
        print('Train set size:', train_set_size)
        batch_size = FLAGS.batch_size
        print('Batch size:', batch_size)
        batch_indexes = [[i, i + batch_size] for i in range(0, train_set_size, batch_size)]
        # print('Batch indexes', batch_indexes)

        for epoch in range(FLAGS.max_epochs):
            shuffle(batch_indexes)
            for batch in batch_indexes:
                sess.run(
                        train_op,
                        feed_dict={
                            i: train_set['features'][batch[0]:batch[1]],
                            o: train_set['targets'] [batch[0]:batch[1]]
                        }
                )

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

        # print('Test features')
        # print(test_set['features'])
        # print('Test targets')
        print('Shape of targets:', test_set['targets'].shape)
        # print(test_set['targets'])
        print('Predictions')
        p_o_i = sess.run(p_o_i, feed_dict={i: test_set['features'], o: test_set['targets']})
        p_o_i_argmax = np.argmax(p_o_i, 3)
        print('Shape of predictions:', p_o_i.shape)
        print('Argmax predictions')
        # print(p_o_i_argmax)
        print()
        for i in range(p_o_i_argmax.shape[0]):
            print('Conversation', i)

            for j in range(p_o_i_argmax.shape[1]):
                c = []
                for k in range(test_set['features'].shape[2]):
                    w = idx2word[test_set['features'][i, j, k]]
                    if w not in ['_SOS_', '_EOS_']:
                        c.append(w)

                s = []
                for k in range(p_o_i_argmax.shape[2]):
                    w = idx2word[p_o_i_argmax[i, j, k]]
                    if w not in ['_SOS_', '_EOS_']:
                        s.append(w)

                print('T {j}: {c:80} |> {s}'.format(j=j, c=' '.join(c), s=' '.join(s)))
            print()


def main(_):
    train_set, test_set, idx2word, word2idx = dataset.dataset()

    train(train_set, test_set, idx2word, word2idx)


if __name__ == '__main__':
    tf.app.run()
