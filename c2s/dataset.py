#!/usr/bin/env python3
import json
from copy import deepcopy

import numpy as np


def load(file_name):
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
            elif mode == 'e2e':
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

    return utterance.upper().split()


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
    words = set()

    for history, target in examples:
        for utterance in history:
            words.update(utterance)
        words.update(target)

    idx2word = ['_SOS_', '_EOS_', '_OOV_']
    idx2word.extend(sorted(words))

    return idx2word


def index_and_pad_utterance(utterance, word2idx, max_length, add_sos=True):
    if add_sos:
        s = [word2idx['_SOS_']]
    else:
        s = []

    for w in utterance:
        s.append(word2idx.get(w, word2idx['_OOV_']))

    for w in range(max_length - len(s)):
        s.append(word2idx['_EOS_'])

    return s


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

    return index_pad_history


def index_and_pad_examples(examples, word2idx, max_length_history, max_length_utterance, max_length_target):
    index_pad_examples = []
    for history, target in examples:
        ip_history = index_and_pad_history(history, word2idx, max_length_history, max_length_utterance)
        ip_target = index_and_pad_utterance(target, word2idx, max_length_target, add_sos=False)

        index_pad_examples.append([ip_history, ip_target])

    return index_pad_examples


def dataset(mode):
    text_data = load('./data.json')

    # examples = gen_examples(text_data, mode='e2e')
    examples = gen_examples(text_data, mode)

    norm_examples = normalize(examples)
    norm_examples = sort_by_conversation_length(norm_examples)

    # print(norm_examples)

    idx2word = get_idx2word(norm_examples)
    word2idx = get_word2idx(idx2word)

    # print(idx2word)
    # print(word2idx)

    max_length_history = 0
    max_length_utterance = 0
    max_length_target = 0
    for history, target in norm_examples:
        for utterance in history:
            max_length_utterance = max(max_length_utterance, len(utterance))

        max_length_history = max(max_length_history, len(history))
        max_length_target = max(max_length_target, len(target))

    # pad the data with _SOS_ and _EOS_ word symbols
    index_examples = index_and_pad_examples(
            norm_examples, word2idx, max_length_history, max_length_utterance, max_length_target
    )

    # print(index_examples)

    # for history, target in index_examples:
    #     for utterance in history:
    #         print('U', len(utterance), utterance)
    #     print('T', len(target), target)

    train_features = [history for history, _ in index_examples]
    train_targets = [target for _, target in index_examples]

    # print(train_features)
    # print(train_targets)

    train_features = np.asarray(train_features, dtype=np.int32)
    train_targets = np.asarray(train_targets, dtype=np.int32)

    # print(train_features)
    # print(train_targets)

    test_features = train_features
    test_targets = train_targets

    train_set = {
        'features': train_features,
        'targets':  train_targets
    }
    test_set = {
        'features': test_features,
        'targets':  test_targets
    }

    return train_set, test_set, idx2word, word2idx
