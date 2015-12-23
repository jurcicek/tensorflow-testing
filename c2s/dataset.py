#!/usr/bin/env python3

import numpy as np

# the text data is composed of conversations and each is composed of questions and answers
text_data = [
    [
        ('Hello, how can I help you?', ''),
        ('I am looking for a restaurant', 'restaurant'),
        ('Ok, you want a restaurant. What food should should serve?', 'restaurant'),
        ('Chinese and it should be in a cheap pricerange', 'restaurant chinese'),
        ('Ok, you want a restaurant serving chinese food in a cheap price range.', 'restaurant chinese cheap'),
    ],
    [
        ('Greetings, how may I help you?', ''),
        ('I am looking for a restaurant serving Indian food no french food', 'restaurant french'),
        ('In what what pricerange?', 'restaurant french'),
        ('Something mid-priced please', 'restaurant french mid-priced'),
    ],
    [
        ('Greetings, how may I help you?', ''),
        ('I am looking for a cheap bar in the city centre', 'bar cheap centre'),
        ('There is no cheap bar in the city centre', 'bar cheap centre'),
        ('Ok, then I do not care about the pricerange.', 'bar centre'),
    ]
]


def get_words(sentence):
    for c in '?!.,':
        sentence = sentence.replace(c, ' ').replace('  ', ' ')

    return sentence.upper().split()


def normalise(text_data):
    text_data_norm = []
    for c in text_data:
        cn = []
        for q, a in c:
            q_words = get_words(q)
            a_words = get_words(a)

            cn.append((q_words, a_words))

        text_data_norm.append(cn)
    return text_data_norm


def get_word2idx(idx2word):
    return dict([(w, i) for i, w in enumerate(idx2word)])


def get_idx2word(text_data):
    words = set(['_OOV_', '_SOS_', '_EOS_'])

    for c in text_data:
        for q, a in c:
            words.update(q)
            words.update(a)

    idx2word = sorted(words)

    return idx2word


def index_and_pad_sentence(sentence, word2idx, max_length, add_sos=True):
    if add_sos:
        s = [word2idx['_SOS_']]
    else:
        s = []
    for w in sentence:
        s.append(word2idx.get(w, word2idx['_OOV_']))

    for w in range(max_length - len(s)):
        s.append(word2idx['_EOS_'])

    return s


def index_and_pad_conversation(conv_data, word2idx, max_length_c, max_length_q, max_length_a):
    conv_data_pad = []

    # padding
    for i in range(max_length_c - len(conv_data)):
        qi = index_and_pad_sentence('', word2idx, max_length_q + 2)
        ai = index_and_pad_sentence('', word2idx, max_length_a + 1, add_sos=False)

        conv_data_pad.append((qi, ai))

    # the real data
    for q, a in conv_data:
        qi = index_and_pad_sentence(q, word2idx, max_length_q + 2)
        ai = index_and_pad_sentence(a, word2idx, max_length_a + 1, add_sos=False)

        conv_data_pad.append((qi, ai))


    return conv_data_pad


def index_and_pad_data(conv_data, word2idx, max_length_c, max_length_q, max_length_a):
    conv_data_pad = []
    for c in conv_data:
        ci = index_and_pad_conversation(c, word2idx, max_length_c, max_length_q, max_length_a)

        conv_data_pad.append(ci)

    return conv_data_pad


def split_q_and_a(index_data):
    qc = []
    ac = []
    for c in index_data:
        qd = []
        ad = []

        for q, a in c:
            qd.append(q)
            ad.append(a)

        qc.append(qd)
        ac.append(ad)

    return qc, ac


def dataset():
    norm_text_data = normalise(text_data)
    # print(norm_text_data)

    idx2word = get_idx2word(norm_text_data)
    word2idx = get_word2idx(idx2word)

    # print(idx2word)
    # print(word2idx)

    max_length_c = 0
    max_length_q = 0
    max_length_a = 0
    for c in norm_text_data:
        for q, a in c:
            max_length_q = max(max_length_q, len(q))
            max_length_a = max(max_length_a, len(a))

        max_length_c = max(max_length_c, len(c))

    # pad the data with _SOS_ and _EOS_ word symbols
    index_data = index_and_pad_data(norm_text_data, word2idx, max_length_c, max_length_q, max_length_a)

    # print(index_data)

    # for i in index_data:
    #     print(len(i))
    #     for q, a in i:
    #         print(len(q), len(a))

    train_features, train_targets = split_q_and_a(index_data)

    # print(train_features)
    # print(train_targets)

    train_features = np.asarray(train_features, dtype=np.int32)
    train_targets = np.asarray(train_targets, dtype=np.int32)

    print(train_features)
    print(train_targets)

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
