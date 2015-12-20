#!/usr/bin/env python3

import numpy as np

# the text data is composed of questions and answers
text_data = [
    ('What color is the sky', 'blue'),
    ('What color is the sun', 'yellow'),
    ('What color is the moon', 'silver'),
    ('What color is sea', 'blue'),
    ('What color is silver jewelery', 'silver'),
    ('What color is grass', 'green'),
    ('Sky, what color is it', 'blue'),
    ('Sun, what color is it', 'yellow'),
    ('I want chinese', 'chinese'),
    ('I want chinese food', 'chinese'),
    ('Indian food', 'indian'),
    ('french food', 'french'),
    ('french restaurant', 'french'),
    ('chinese restaurant', 'chinese'),
    ('I am looking for a restaurant serving Indian food', 'indian'),
    ('I am looking for a restaurant serving Indian food no french food', 'french'),
    ('Indian no chinese food', 'chinese'),
]


def get_words(sentence):
    for c in '?!.,':
        sentence = sentence.replace(c, ' ').replace('  ', ' ')

    return sentence.upper().split()


def normalise(text_data):
    td = []
    for q, a in text_data:
        q_words = get_words(q)
        a_words = get_words(a)

        td.append((q_words, a_words))

    return td


def get_word2idx(idx2word):
    return dict([(w, i) for i, w in enumerate(idx2word)])


def get_idx2word(text_data):
    words = set(['_OOV_', '_SOS_', '_EOS_'])

    for q, a in text_data:
        words.update(q)
        words.update(a)

    idx2word = sorted(words)

    return idx2word


def transform_sentence(sentence, word2idx, max_length):
    s = [word2idx['_SOS_']]
    for w in sentence:
        s.append(word2idx.get(w, word2idx['_OOV_']))

    for w in range(0, max_length - len(s)):
        s.append(word2idx['_EOS_'])

    return s


def transform_data(text_data, word2idx, max_length_q, max_length_a):
    id = []
    for q, a in text_data:
        qi = transform_sentence(q, word2idx, max_length_q)
        ai = transform_sentence(a, word2idx, max_length_a)

        id.append((qi, ai))

    return id


def split_q_and_a(index_data):
    qd = []
    ad = []

    for q, a in index_data:
        qd.append(q)
        ad.append(a)

    return qd, ad


def dataset():
    norm_text_data = normalise(text_data)
    print(norm_text_data)

    idx2word = get_idx2word(norm_text_data)
    word2idx = get_word2idx(idx2word)

    # print(idx2word)
    # print(word2idx)

    max_length_q = 0
    max_length_a = 0
    for q, a in norm_text_data:
        max_length_q = max(max_length_q, len(q))
        max_length_a = max(max_length_a, len(a))

    # print(max_length)

    # add two for _SOS_ and _EOS_ word symbols
    index_data = transform_data(norm_text_data, word2idx, max_length_q + 2, max_length_a + 2)

    print(index_data)

    train_features, train_targets = split_q_and_a(index_data)

    train_features = np.asarray(train_features, dtype=np.int32)
    train_targets = np.asarray(train_targets, dtype=np.int32)

    test_features = train_features
    test_targets = train_targets

    train_set = {
        'features': train_features,
        'targets':  train_targets[:, 1].reshape((-1, 1))
    }
    test_set = {
        'features': test_features,
        'targets':  test_targets[:, 1].reshape((-1, 1))
    }

    print(train_targets)
    print(train_set['targets'])

    return train_set, test_set, idx2word, word2idx
