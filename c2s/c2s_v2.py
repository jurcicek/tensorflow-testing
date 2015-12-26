#!/usr/bin/env python3
import sys

sys.path.extend(['..'])

import numpy as np
import tensorflow as tf

from tensorflow.python.ops.rnn_cell import LSTMCell
from random import shuffle

import dataset

from tf_ext.bricks import embedding, rnn, rnn_decoder, dense_to_one_hot, brnn
from tf_ext.optimizers import AdamPlusOptimizer

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_epochs', 1000, 'Number of epochs to run trainer.')
flags.DEFINE_integer('batch_size', 2, 'Number of training examples in a batch.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('decay', 0.9, 'AdamPlusOptimizer learning rate decay.')
flags.DEFINE_float('beta1', 0.01, 'AdamPlusOptimizer 1st moment decay.')
flags.DEFINE_float('beta2', 0.999, 'AdamPlusOptimizer 2nd moment decay.')
flags.DEFINE_float('epsilon', 1e-5, 'AdamPlusOptimizer epsilon.')
flags.DEFINE_float('pow', 0.7, 'AdamPlusOptimizer pow.')
flags.DEFINE_float('regularization', 1e-3, 'Weight of regularization.')
flags.DEFINE_float('max_gradient_norm', 5e0, 'Clip gradients to this norm.')
flags.DEFINE_boolean('print_variables', False, 'Print all trainable variables.')

"""
This code shows how to build and train a sequence to answer translation model.
This model build bidirectional RNN layers before encoding a turn or a conversation using forward RNN.
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

        with tf.name_scope("TurnsEncoder"):
            with tf.name_scope("RNNForwardTurnEncoderCell_1"):
                cell_fw_1 = LSTMCell(
                        num_units=encoder_lstm_size,
                        input_size=encoder_embedding_size,
                        use_peepholes=True
                )
                initial_state_fw_1 = cell_fw_1.zero_state(batch_size, tf.float32)

            with tf.name_scope("RNNBackwardTurnEncoderCell_1"):
                cell_bw_1 = LSTMCell(
                        num_units=encoder_lstm_size,
                        input_size=encoder_embedding_size,
                        use_peepholes=True
                )
                initial_state_bw_1 = cell_bw_1.zero_state(batch_size, tf.float32)

            with tf.name_scope("RNNForwardTurnEncoderCell_2"):
                cell_fw_2 = LSTMCell(
                        num_units=encoder_lstm_size,
                        input_size=cell_fw_1.output_size + cell_bw_1.output_size,
                        use_peepholes=True
                )
                initial_state_fw_2 = cell_fw_2.zero_state(batch_size, tf.float32)

            # the input data has this dimensions
            # [#batch, #turn (sentence) in a conversation (a dialogue), #word in a turn (a sentence), # embedding dimension]

            # encode all turns along the word axis
            encoder_states_2d = []

            for turn in range(conversation_length):
                encoder_outputs, _ = brnn(
                        cell_fw=cell_fw_1,
                        cell_bw=cell_bw_1,
                        inputs=[encoder_embedding[:, turn, word, :] for word in range(encoder_sequence_length)],
                        initial_state_fw=initial_state_fw_1,
                        initial_state_bw=initial_state_bw_1,
                        name='RNNTurnBidirectionalLayer',
                        reuse=True if turn > 0 else None
                )

                _, encoder_states = rnn(
                        cell=cell_fw_2,
                        inputs=encoder_outputs,
                        initial_state=initial_state_fw_2,
                        name='RNNTurnForwardEncoder',
                        reuse=True if turn > 0 else None
                )

                # print(encoder_states[-1])
                encoder_states = tf.concat(1, tf.expand_dims(encoder_states[-1], 1))
                # print(encoder_states)
                encoder_states_2d.append(encoder_states)

            encoder_states_2d = tf.concat(1, encoder_states_2d)
            # print('encoder_states_2d', encoder_states_2d)

        with tf.name_scope("ConversationEncoder"):
            # encode all conversations along the turn axis
            with tf.name_scope("RNNFrowardConversationEncoderCell_1"):
                cell_fw_1 = LSTMCell(
                        num_units=encoder_lstm_size,
                        input_size=cell_fw_2.state_size,
                        use_peepholes=True
                )
                initial_state_fw_1 = cell_fw_1.zero_state(batch_size, tf.float32)

            with tf.name_scope("RNNBackwardConversationEncoderCell_1"):
                cell_bw_1 = LSTMCell(
                        num_units=encoder_lstm_size,
                        input_size=cell_fw_2.state_size,
                        use_peepholes=True
                )
                initial_state_bw_1 = cell_fw_2.zero_state(batch_size, tf.float32)

            with tf.name_scope("RNNFrowardConversationEncoderCell_2"):
                cell_fw_2 = LSTMCell(
                        num_units=encoder_lstm_size,
                        input_size=cell_fw_1.output_size + cell_bw_1.output_size,
                        use_peepholes=True
                )
                initial_state_fw_2 = cell_fw_2.zero_state(batch_size, tf.float32)

            encoder_outputs, _ = brnn(
                    cell_fw=cell_fw_1,
                    cell_bw=cell_bw_1,
                    inputs=[encoder_states_2d[:, turn, :] for turn in range(conversation_length)],
                    initial_state_fw=initial_state_fw_1,
                    initial_state_bw=initial_state_bw_1,
                    name='RNNConversationBidirectionalLayer',
                    reuse=None
            )

            _, encoder_states = rnn(
                    cell=cell_fw_2,
                    inputs=encoder_outputs,
                    initial_state=initial_state_fw_2,
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
            for n in ['/W_', '/W:', '/B:']:
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

        tvars = tf.trainable_variables()
        learning_rate = tf.Variable(float(FLAGS.learning_rate), trainable=False)

        train_op = AdamPlusOptimizer(
                learning_rate=learning_rate,
                beta1=FLAGS.beta1,
                beta2=FLAGS.beta2,
                epsilon=FLAGS.epsilon,
                pow=FLAGS.pow,
                use_locking=True,
                name='trainer')

        learning_rate_decay_op = learning_rate.assign(learning_rate * FLAGS.decay)
        global_step = tf.Variable(0, trainable=False)
        gradients = tf.gradients(loss, tvars)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
        # clipped_gradients = [tf.clip_by_norm(g, FLAGS.max_gradient_norm) for g in gradients]
        train_op = train_op.apply_gradients(zip(clipped_gradients, tvars), global_step=global_step)

        tf.initialize_all_variables().run()

        # prepare batch indexes
        train_set_size = train_set['features'].shape[0]
        print('Train set size:', train_set_size)
        batch_size = FLAGS.batch_size
        print('Batch size:', batch_size)
        batch_indexes = [[i, i + batch_size] for i in range(0, train_set_size, batch_size)]
        # print('Batch indexes', batch_indexes)

        previous_accuracies = []
        previous_losses = []
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

            if epoch % max(min(int(FLAGS.max_epochs / 100), 100), 1) == 0:
                summary, lss, acc = sess.run([merged, loss, accuracy],
                                             feed_dict={i: test_set['features'], o: test_set['targets']})
                writer.add_summary(summary, epoch)
                print()
                print('Epoch: {epoch}'.format(epoch=epoch))
                print(' - accuracy      = {acc:f}'.format(acc=acc))
                print(' - loss          = {lss:f}'.format(lss=lss))
                print(' - learning rate = {lr:f}'.format(lr=learning_rate.eval()))

                # decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and lss > max(previous_losses[-3:]):
                    sess.run(learning_rate_decay_op)
                previous_losses.append(lss)

                # stop when reached a threshold maximum or when no improvement in the last 20 steps
                previous_accuracies.append(acc)
                if acc > 0.9999 or max(previous_accuracies) > max(previous_accuracies[-20:]):
                    break

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
