#!/usr/bin/env python3
import json
from collections import defaultdict
from copy import deepcopy
from random import shuffle

import numpy as np
import sys

import ontology

def load_json_data(file_name):
    """Load a json file - file_name.

    :param file_name: a name of the json file
    :return: the Python representation of the json file
    """
    with open(file_name) as f:
        text_data = json.load(f)

    return text_data


def gen_examples(text_data, mode='tracker'):
    """Generates training examples for the conversation to sequence model. Here, we experiment with conversational models
    that is converting a conversational history into dialogue state representation (dialogue state tracking) or generation
    a textual response given the conversation history (dialogue policy).

    :param text_data: a list of conversation each composed of (system_output, user_input, dialogue state) tuples
    :param mode: "tracker" - generate examples for dialogue state tracking,
                 "e2e      - e2e generates examples word to word dialogue management
    :return: a transformed text_data into a series of training/development/testing examples
    """
    examples = []
    for conversation in text_data:
        history = []
        prev_turn = None
        for turn in conversation:
            target = None
            if mode == 'tracker':
                history.append(turn[0])
                history.append(turn[1])
                target = turn[2]  # the dialogue state
            else:  # mode == 'e2e'
                if prev_turn:
                    history.append(prev_turn[0])
                    history.append(prev_turn[1])
                    target = turn[0]  # the dialogue state

                prev_turn = turn

            # print(history)
            # print(target)

            if target:
                examples.append(deepcopy([history, target]))

    return examples


def get_words(utterance):
    """Splits an utterance into words, removes some characters not available in spoken dialogue systems,
    uppercases the text.

    :param utterance: a string
    :return: a list of string (words)
    """
    for c in '?!.,':
        utterance = utterance.replace(c, ' ').replace('  ', ' ')

    return utterance.lower().split()


def normalize(examples):
    norm_examples = []
    for history, target in examples:
        norm_history = []
        for utterance in history:
            utterance_words = get_words(utterance)
            norm_history.append(utterance_words)

        norm_target = get_words(target)

        norm_examples.append([norm_history, norm_target])

    return norm_examples


def sort_by_conversation_length(examples):
    examples.sort(key=lambda example: len(example[0]))

    return examples


def get_word2idx(idx2word):
    return dict([(w, i) for i, w in enumerate(idx2word)])


def get_idx2word(examples):
    words_history = defaultdict(int)
    words_target = defaultdict(int)

    for history, target in examples:
        for utterance in history:
            for word in utterance:
                words_history[word] += 1

        for word in target:
            words_target[word] += 1

    idx2word_history = ['_SOS_', '_EOS_', '_OOV_']
    words_history = [word for word in words_history if words_history[word] >= 3]
    idx2word_history.extend(sorted(words_history))

    idx2word_target = ['_SOS_', '_EOS_', '_OOV_']
    words_target = [word for word in words_history if words_target[word] >= 3]
    idx2word_target.extend(sorted(words_target))

    return idx2word_history, idx2word_target


def index_and_pad_utterance(utterance, word2idx, max_length, add_sos=True):
    if add_sos:
        s = [word2idx['_SOS_']]
    else:
        s = []

    for w in utterance:
        if w not in word2idx:
            print('OOV: {oov}'.format(oov=w))
        s.append(word2idx.get(w, word2idx['_OOV_']))

    for w in range(max_length - len(s)):
        s.append(word2idx['_EOS_'])

    return s[:max_length]


def index_and_pad_history(history, word2idx, max_length_history, max_length_utterance):
    index_pad_history = []

    # padding
    for i in range(max_length_history - len(history)):
        ip_utterance = index_and_pad_utterance('', word2idx, max_length_utterance + 2)

        index_pad_history.append(ip_utterance)

    # the real data
    for utterance in history:
        ip_utterance = index_and_pad_utterance(utterance, word2idx, max_length_utterance + 2)

        index_pad_history.append(ip_utterance)

    return index_pad_history[len(index_pad_history) - max_length_history:]


def index_and_pad_examples(examples, word2idx_history, max_length_history, max_length_utterance,
                           word2idx_target, max_length_target):
    index_pad_examples = []
    for history, target in examples:
        ip_history = index_and_pad_history(history, word2idx_history, max_length_history, max_length_utterance)
        ip_target = index_and_pad_utterance(target, word2idx_target, max_length_target, add_sos=False)

        index_pad_examples.append([ip_history, ip_target])

    return index_pad_examples


class DSTC2:
    def __init__(self, mode, train_data_fn, data_fraction, test_data_fn, ontology_fn, database_fn, batch_size):
        self.ontology = ontology.Ontology(ontology_fn, database_fn)

        train_data = load_json_data(train_data_fn)
        train_data = train_data[:int(len(train_data) * min(data_fraction, 1.0))]
        test_data = load_json_data(test_data_fn)
        test_data = test_data[:int(len(test_data) * min(data_fraction, 1.0))]

        train_examples = gen_examples(train_data, mode)
        test_examples = gen_examples(test_data, mode)
        # print(train_examples)

        norm_train_examples = normalize(train_examples)
        norm_train_examples = sort_by_conversation_length(norm_train_examples)
        # remove 10 % of the longest dialogues this will half the length of the conversations
        norm_train_examples = norm_train_examples[:-int(len(norm_train_examples) / 10)]
        # print(norm_train_examples)

        norm_test_examples = normalize(test_examples)
        norm_test_examples = sort_by_conversation_length(norm_test_examples)
        # remove 10 % of the longest dialogues
        norm_test_examples = norm_test_examples[:-int(len(norm_test_examples) / 10)]

        abstract_train_examples, arguments_train_examples = self.ontology.abstract(norm_train_examples, mode)
        abstract_test_examples, arguments_test_examples = self.ontology.abstract(norm_test_examples, mode)

        for history, target in abstract_train_examples:
            print('-'*120)
            for utterance in history:
                print('U', ' '.join(utterance))
                print('T', ' '.join(target))
        for history, target in abstract_test_examples:
            print('-'*120)
            for utterance in history:
                print('U', ' '.join(utterance))
                print('T', ' '.join(target))
        # sys.exit(0)

        idx2word_history, idx2word_target = get_idx2word(abstract_train_examples)
        word2idx_history = get_word2idx(idx2word_history)
        word2idx_target = get_word2idx(idx2word_target)

        # print(idx2word_history)
        # print(word2idx_history)
        # print()
        # print(idx2word_target)
        # print(word2idx_target)

        max_length_history = 0
        max_length_utterance = 0
        max_length_target = 0
        for history, target in abstract_train_examples:
            for utterance in history:
                max_length_utterance = max(max_length_utterance, len(utterance))

            max_length_history = max(max_length_history, len(history))
            max_length_target = max(max_length_target, len(target))

        # pad the data with _SOS_ and _EOS_ word symbols
        train_index_examples = index_and_pad_examples(
                abstract_train_examples, word2idx_history, max_length_history, max_length_utterance,
                word2idx_target, max_length_target
        )
        # print(train_index_examples)

        # for history, target in train_index_examples:
        #     for utterance in history:
        #         print('U', len(utterance), utterance)
        #     print('T', len(target), target)

        train_features = [history for history, _ in train_index_examples]
        train_targets = [target for _, target in train_index_examples]

        train_features = np.asarray(train_features, dtype=np.int32)
        train_targets = np.asarray(train_targets, dtype=np.int32)

        # print(train_features)
        # print(train_targets)

        test_index_examples = index_and_pad_examples(
                abstract_test_examples, word2idx_history, max_length_history, max_length_utterance,
                word2idx_target, max_length_target
        )
        # print(test_index_examples)

        # for history, target in test_index_examples:
        #     for utterance in history:
        #         print('U', len(utterance), utterance)
        #     print('T', len(target), target)

        test_features = [history for history, _ in test_index_examples]
        test_targets = [target for _, target in test_index_examples]

        test_features = np.asarray(test_features, dtype=np.int32)
        test_targets = np.asarray(test_targets, dtype=np.int32)

        # print(test_features)
        # print(test_targets)

        train_set = {
            'features': train_features,
            'targets':  train_targets
        }
        dev_set = {
            'features': train_features[int(len(train_set) * 0.9):],
            'targets': train_targets  [int(len(train_set) * 0.9):]
        }
        test_set = {
            'features': test_features,
            'targets':  test_targets
        }

        self.train_set = train_set
        self.train_set_size = len(train_set['features'])
        self.dev_set = dev_set
        self.dev_set_size = len(dev_set['features'])
        self.test_set = test_set
        self.test_set_size = len(test_set['features'])

        self.idx2word_history = idx2word_history
        self.word2idx_history = word2idx_history
        self.idx2word_target = idx2word_target
        self.word2idx_target = word2idx_target

        self.batch_size = batch_size
        self.train_batch_indexes = [[i, i + batch_size] for i in range(0, self.train_set_size, batch_size)]

    def iter_train_batches(self):
        for batch in self.train_batch_indexes:
            yield {
                'features': self.train_set['features'][batch[0]:batch[1]],
                'targets': self.train_set['targets']  [batch[0]:batch[1]],
            }
        shuffle(self.train_batch_indexes)
