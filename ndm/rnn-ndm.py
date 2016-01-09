#!/usr/bin/env python3
import sys

sys.path.extend(['..'])

import numpy as np
import tensorflow as tf

from tensorflow.python.ops.rnn_cell import LSTMCell
from random import shuffle

import dataset

from tf_ext.bricks import embedding, rnn, rnn_decoder, dense_to_one_hot, brnn, device_for_node_cpu, \
    device_for_node_gpu_matmul
from tf_ext.optimizers import AdamPlusOptimizer

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('task', 'tracker', '"tracker" (dialogue state tracker) | '
                                       '"e2e" (word to word dialogue management)')
flags.DEFINE_string('data', './data.handcrafted.json', 'The data the model should be trained on.')
flags.DEFINE_integer('max_epochs', 1000, 'Number of epochs to run trainer.')
flags.DEFINE_integer('batch_size', 100, 'Number of training examples in a batch.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('decay', 0.9, 'AdamPlusOptimizer learning rate decay.')
flags.DEFINE_float('beta1', 0.9, 'AdamPlusOptimizer 1st moment decay.')
flags.DEFINE_float('beta2', 0.999, 'AdamPlusOptimizer 2nd moment decay.')
flags.DEFINE_float('epsilon', 1e-5, 'AdamPlusOptimizer epsilon.')
flags.DEFINE_float('pow', 0.9, 'AdamPlusOptimizer pow.')
flags.DEFINE_float('regularization', 1e-6, 'Weight of regularization.')
flags.DEFINE_float('max_gradient_norm', 5e0, 'Clip gradients to this norm.')
flags.DEFINE_float('use_inputs_prob_decay', 0.999, 'Decay of the probability of using '
                                                   'the true targets during generation.')
flags.DEFINE_boolean('print_variables', False, 'Print all trainable variables.')

"""
This code shows how to build and train a sequence to answer translation model.
This model builds bidirectional RNN layers before encoding an utterance or a history using forward RNN.
"""


def train(train_set, test_set, idx2word_history, word2idx_history, idx2word_target, word2idx_target):
    with tf.variable_scope("history_length"):
        history_length = train_set['features'].shape[1]

    encoder_lstm_size = 16*4
    encoder_embedding_size = 16*8
    encoder_vocabulary_length = len(idx2word_history)
    with tf.variable_scope("encoder_sequence_length"):
        encoder_sequence_length = train_set['features'].shape[2]

    decoder_lstm_size = 16*4
    decoder_embedding_size = 16*4
    decoder_vocabulary_length = len(idx2word_target)
    with tf.variable_scope("decoder_sequence_length"):
        decoder_sequence_length = train_set['targets'].shape[1]

    # inference model
    with tf.name_scope('model'):
        features = tf.placeholder("int32", name='features')
        targets = tf.placeholder("int32", name='true_targets')
        use_dropout_prob = tf.placeholder("float32", name='use_dropout_prob')

        with tf.variable_scope("batch_size"):
            batch_size = tf.shape(features)[0]

        encoder_embedding = embedding(
                input=features,
                length=encoder_vocabulary_length,
                size=encoder_embedding_size,
                name='encoder_embedding'
        )

        with tf.name_scope("UtterancesEncoder"):
            with tf.name_scope("RNNForwardUtteranceEncoderCell_1"):
                cell_fw_1 = LSTMCell(
                        num_units=encoder_lstm_size,
                        input_size=encoder_embedding_size,
                        use_peepholes=True
                )
                initial_state_fw_1 = cell_fw_1.zero_state(batch_size, tf.float32)

            with tf.name_scope("RNNBackwardUtteranceEncoderCell_1"):
                cell_bw_1 = LSTMCell(
                        num_units=encoder_lstm_size,
                        input_size=encoder_embedding_size,
                        use_peepholes=True
                )
                initial_state_bw_1 = cell_bw_1.zero_state(batch_size, tf.float32)

            with tf.name_scope("RNNForwardUtteranceEncoderCell_2"):
                cell_fw_2 = LSTMCell(
                        num_units=encoder_lstm_size,
                        input_size=cell_fw_1.output_size + cell_bw_1.output_size,
                        use_peepholes=True
                )
                initial_state_fw_2 = cell_fw_2.zero_state(batch_size, tf.float32)

            # the input data has this dimensions
            # [
            #   #batch,
            #   #utterance in a history (a dialogue),
            #   #word in an utterance (a sentence),
            #   embedding dimension
            # ]

            # encode all utterances along the word axis
            encoder_states_2d = []

            for utterance in range(history_length):
                encoder_outputs, _ = brnn(
                        cell_fw=cell_fw_1,
                        cell_bw=cell_bw_1,
                        inputs=[encoder_embedding[:, utterance, word, :] for word in range(encoder_sequence_length)],
                        initial_state_fw=initial_state_fw_1,
                        initial_state_bw=initial_state_bw_1,
                        name='RNNUtteranceBidirectionalLayer',
                        reuse=True if utterance > 0 else None
                )

                _, encoder_states = rnn(
                        cell=cell_fw_2,
                        inputs=encoder_outputs,
                        initial_state=initial_state_fw_2,
                        name='RNNUtteranceForwardEncoder',
                        reuse=True if utterance > 0 else None
                )

                # print(encoder_states[-1])
                encoder_states = tf.concat(1, tf.expand_dims(encoder_states[-1], 1))
                # print(encoder_states)
                encoder_states_2d.append(encoder_states)

            encoder_states_2d = tf.concat(1, encoder_states_2d)
            # print('encoder_states_2d', encoder_states_2d)

        with tf.name_scope("HistoryEncoder"):
            # encode all histories along the utterance axis
            with tf.name_scope("RNNFrowardHistoryEncoderCell_1"):
                cell_fw_1 = LSTMCell(
                        num_units=encoder_lstm_size,
                        input_size=cell_fw_2.state_size,
                        use_peepholes=True
                )
                initial_state_fw_1 = cell_fw_1.zero_state(batch_size, tf.float32)

            with tf.name_scope("RNNBackwardHistoryEncoderCell_1"):
                cell_bw_1 = LSTMCell(
                        num_units=encoder_lstm_size,
                        input_size=cell_fw_2.state_size,
                        use_peepholes=True
                )
                initial_state_bw_1 = cell_fw_2.zero_state(batch_size, tf.float32)

            with tf.name_scope("RNNFrowardHistoryEncoderCell_2"):
                cell_fw_2 = LSTMCell(
                        num_units=encoder_lstm_size,
                        input_size=cell_fw_1.output_size + cell_bw_1.output_size,
                        use_peepholes=True
                )
                initial_state_fw_2 = cell_fw_2.zero_state(batch_size, tf.float32)

            encoder_outputs, _ = brnn(
                    cell_fw=cell_fw_1,
                    cell_bw=cell_bw_1,
                    inputs=[encoder_states_2d[:, utterance, :] for utterance in range(history_length)],
                    initial_state_fw=initial_state_fw_1,
                    initial_state_bw=initial_state_bw_1,
                    name='RNNHistoryBidirectionalLayer',
                    reuse=None
            )

            _, encoder_states = rnn(
                    cell=cell_fw_2,
                    inputs=encoder_outputs,
                    initial_state=initial_state_fw_2,
                    name='RNNHistoryForwardEncoder',
                    reuse=None
            )

        with tf.name_scope("Decoder"):
            use_inputs_prob = tf.Variable(1.0, name='use_inputs_prob', trainable=False)
            use_inputs_prob_decay_op = use_inputs_prob.assign(use_inputs_prob * FLAGS.use_inputs_prob_decay)

            with tf.name_scope("RNNDecoderCell"):
                cell = LSTMCell(
                        num_units=decoder_lstm_size,
                        input_size=decoder_embedding_size,
                        use_peepholes=True,
                )

            # decode all histories along the utterance axis
            final_encoder_state = encoder_states[-1]

            decoder_states, decoder_outputs, decoder_outputs_softmax = rnn_decoder(
                    cell=cell,
                    inputs=[targets[:, word] for word in range(decoder_sequence_length)],
                    initial_state=final_encoder_state,
                    embedding_size=decoder_embedding_size,
                    embedding_length=decoder_vocabulary_length,
                    sequence_length=decoder_sequence_length,
                    name='RNNDecoder',
                    reuse=False,
                    use_inputs_prob=use_inputs_prob
            )

            targets_give_features = tf.concat(1, decoder_outputs_softmax)
            # print(p_o_i)

    if FLAGS.print_variables:
        for v in tf.trainable_variables():
            print(v.name)

    with tf.name_scope('loss'):
        one_hot_labels = dense_to_one_hot(targets, decoder_vocabulary_length)
        loss = tf.reduce_mean(- one_hot_labels * tf.log(targets_give_features), name='loss')
        for v in tf.trainable_variables():
            for n in ['/W_', '/W:', '/B:']:
                if n in v.name:
                    print('Regularization using', v.name)
                    loss += FLAGS.regularization * tf.reduce_mean(tf.pow(v, 2))
        tf.scalar_summary('loss', loss)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(one_hot_labels, 2), tf.argmax(targets_give_features, 2))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        tf.scalar_summary('accuracy', accuracy)

    # with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    with tf.Session() as sess:
        # Merge all the summaries and write them out to ./log
        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter('./log', sess.graph_def)
        saver = tf.train.Saver()

        # training
        tvars = tf.trainable_variables()
        # tvars = [v for v in tvars if 'embedding_table' not in v.name] # all variables except embeddings
        learning_rate = tf.Variable(float(FLAGS.learning_rate), trainable=False)

        # train_op = tf.train.GradientDescentOptimizer(
        train_op = AdamPlusOptimizer(
                learning_rate=learning_rate,
                beta1=FLAGS.beta1,
                beta2=FLAGS.beta2,
                epsilon=FLAGS.epsilon,
                pow=FLAGS.pow,
                use_locking=False,
                name='trainer')

        learning_rate_decay_op = learning_rate.assign(learning_rate * FLAGS.decay)
        global_step = tf.Variable(0, trainable=False)
        gradients = tf.gradients(loss, tvars)

        clipped_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
        train_op = train_op.apply_gradients(zip(clipped_gradients, tvars), global_step=global_step)

        tf.initialize_all_variables().run()

        # prepare batch indexes
        train_set_size = train_set['features'].shape[0]
        print('Train set size:', train_set_size)
        batch_size = FLAGS.batch_size
        print('Batch size:', batch_size)
        batch_indexes = [[i, i + batch_size] for i in range(0, train_set_size, batch_size)]
        print('#Batches:', len(batch_indexes))
        # print('Batch indexes', batch_indexes)

        previous_accuracies = []
        previous_losses = []
        for epoch in range(FLAGS.max_epochs):
            print('Batch: ', end=' ', flush=True)
            for b, batch in enumerate(batch_indexes):
                print(b, end=' ', flush=True)
                sess.run(
                        train_op,
                        feed_dict={
                            features: train_set['features'][batch[0]:batch[1]],
                            targets: train_set['targets'][batch[0]:batch[1]],
                        }
                )
            print()
            shuffle(batch_indexes)

            if epoch % max(min(int(FLAGS.max_epochs / 100), 100), 1) == 0:
                summary, lss, acc = sess.run([merged, loss, accuracy],
                                             feed_dict={features: test_set['features'], targets: test_set['targets']})
                writer.add_summary(summary, epoch)
                print()
                print('Epoch: {epoch}'.format(epoch=epoch))
                print(' - accuracy        = {acc:f}'.format(acc=acc))
                print(' - loss            = {lss:f}'.format(lss=lss))
                print(' - learning rate   = {lr:f}'.format(lr=learning_rate.eval()))
                print(' - use inputs prob = {uip:f}'.format(uip=use_inputs_prob.eval()))
                print()

                # decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and lss > max(previous_losses[-3:]):
                    sess.run(learning_rate_decay_op)
                previous_losses.append(lss)

                # stop when reached a threshold maximum or when no improvement in the last 20 steps
                previous_accuracies.append(acc)
                if acc > 0.9999 or max(previous_accuracies) > max(previous_accuracies[-20:]):
                    break

            sess.run(use_inputs_prob_decay_op)

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
        targets_give_features = sess.run(targets_give_features,
                                         feed_dict={features: test_set['features'], targets: test_set['targets']})
        targets_given_features_argmax = np.argmax(targets_give_features, 2)
        print('Shape of predictions:', targets_give_features.shape)
        print('Argmax predictions')
        # print(p_o_i_argmax)
        print()
        for features in range(0, targets_given_features_argmax.shape[0], max(int(targets_given_features_argmax.shape[0]/10), 1)):
            print('History', features)

            for j in range(test_set['features'].shape[1]):
                utterance = []
                for k in range(test_set['features'].shape[2]):
                    w = idx2word_history[test_set['features'][features, j, k]]
                    if w not in ['_SOS_', '_EOS_']:
                        utterance.append(w)
                print('U {j}: {c:80}'.format(j=j, c=' '.join(utterance)))

            prediction = []
            for j in range(targets_given_features_argmax.shape[1]):
                w = idx2word_target[targets_given_features_argmax[features, j]]
                if w not in ['_SOS_', '_EOS_']:
                    prediction.append(w)

            print('P  : {t:80}'.format(t=' '.join(prediction)))

            target = []
            for j in range(test_set['targets'].shape[1]):
                w = idx2word_target[test_set['targets'][features, j]]
                if w not in ['_SOS_', '_EOS_']:
                    target.append(w)

            print('T  : {t:80}'.format(t=' '.join(target)))
            print()


def main(_):
    graph = tf.Graph()

    with graph.as_default():
        with graph.device(device_for_node_cpu):
            print('-' * 120)
            print('RNN-nDM')
            print('    task                  = {t}'.format(t=FLAGS.task))
            print('    data                  = {data}'.format(data=FLAGS.data))
            print('    max_epochs            = {max_epochs}'.format(max_epochs=FLAGS.max_epochs))
            print('    batch_size            = {batch_size}'.format(batch_size=FLAGS.batch_size))
            print('    learning_rate         = {learning_rate}'.format(learning_rate=FLAGS.learning_rate))
            print('    decay                 = {decay}'.format(decay=FLAGS.decay))
            print('    beta1                 = {beta1}'.format(beta1=FLAGS.beta1))
            print('    beta2                 = {beta2}'.format(beta2=FLAGS.beta2))
            print('    epsilon               = {epsilon}'.format(epsilon=FLAGS.epsilon))
            print('    pow                   = {pow}'.format(pow=FLAGS.pow))
            print('    regularization        = {regularization}'.format(regularization=FLAGS.regularization))
            print('    max_gradient_norm     = {max_gradient_norm}'.format(max_gradient_norm=FLAGS.max_gradient_norm))
            print('    use_inputs_prob_decay = {use_inputs_prob_decay}'.format(
                    use_inputs_prob_decay=FLAGS.use_inputs_prob_decay))
            print('-' * 120)
            train_set, test_set, idx2word_history, word2idx_history, idx2word_target, word2idx_target = dataset.load(
                    mode=FLAGS.task, text_data_fn=FLAGS.data
            )

            print('Input vocabulary size:  ', len(idx2word_history))
            print('Output vocabulary size: ', len(idx2word_target))
            print('-' * 120)

            train(train_set, test_set, idx2word_history, word2idx_history, idx2word_target, word2idx_target)


if __name__ == '__main__':
    tf.app.run()
