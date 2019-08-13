import tensorflow as tf
from tensorflow import keras
import unicodedata
import re

EOS_token = 1
SOS_token = 0


def unicodeToAscii(string):
    return "".join(c for c in unicodedata.normalize("NFD", string)
                   if unicodedata.category(c) != "Mn")


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def sentenceToIndexes(sentence, language):
    return [language.word2index[word] for word in sentence.split()] + [EOS_token]


class Language(object):
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.index2word = {0: "SOS",
                           1: "EOS"}
        self.word2count = {}
        self.n_words = 2

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word2count[word] = 1
            self.n_words += 1

        else:
            self.word2count[word] += 1

    def addSentence(self, sentence):
        for word in sentence.strip().split():
            self.addWord(word)


def build_lang(lang1, lang2, data_path, max_length=10):
    input_lang = Language(lang1)
    output_lang = Language(lang2)
    input_seq = []
    output_seq = []
    pairs = load_dataset(data_path)

    for pair in pairs:
        input_lang.addSentence(pair[1])
        output_lang.addSentence(pair[0])
    for pair in pairs:
        input_seq.append(sentenceToIndexes(pair[1], input_lang))
        output_seq.append(sentenceToIndexes(pair[0], output_lang))

        # pairs = filterPairs(pairs, 10)
    return keras.preprocessing.sequence.pad_sequences(input_seq, maxlen=max_length, padding='post',
                                                      truncating='post'), \
           keras.preprocessing.sequence.pad_sequences(output_seq, maxlen=max_length, padding='post',
                                                      truncating='post'), input_lang, output_lang, pairs


def load_dataset(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    pairs = [[normalizeString(pair) for pair in
              line.strip().split('\t')] for line in lines]

    return pairs


def loss_fn(real, pred):
    criterion = keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                           reduction='none')
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    _loss = criterion(real, pred)
    mask = tf.cast(mask, dtype=_loss.dtype)
    _loss *= mask
    return tf.reduce_mean(_loss)


def filterPair(p, max_length):
    return len(p[0].split()) < max_length and \
           len(p[1].split()) < max_length


def filterPairs(pairs, max_length):
    return [pair for pair in pairs if filterPair(pair, max_length)]


def checkpoint(model, name=None):
    if name is not None:
        model.save_weights("checkpoint/{}.h5".format(name))
    else:
        raise NotImplementedError
