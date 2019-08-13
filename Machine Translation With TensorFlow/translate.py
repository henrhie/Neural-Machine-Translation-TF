import tensorflow as tf
import numpy as np
import os
import Utils
import train
import argparse

SOS_token = 0

decoder = train.Decoder
encoder = train.Encoder


def translate(args):
    if os.path.exists("checkpoint/encoder.h5") and \
            os.path.exists("checkpoint/decoder.h5"):
        hidden = [tf.zeros((1, 256))]

        # runs forward propagation through the models to build the model
        _ = train.train_step(encoder, decoder, tf.ones((1, 2)),
                             tf.ones((1, 2)), hidden)
        
        encoder.load_weights("checkpoint/encoder.h5")
        decoder.load_weights("checkpoint/decoder.h5")
        print("models loaded....")

    result = ''
    attention_plot = np.zeros((10, 10))
    sentence = Utils.normalizeString(args.sentence)
    sentence = Utils.sentenceToIndexes(sentence, train.input_lang)
    sentence = Utils.keras.preprocessing.sequence.pad_sequences([sentence], padding='post',
                                                                maxlen=args.max_length,
                                                                truncating='post')

    hidden = [tf.zeros((1, 256))]

    enc_out, enc_hidden = encoder(sentence, hidden)

    dec_hidden = enc_hidden
    SOS_tensor = np.array([SOS_token])
    dec_input = tf.squeeze(tf.expand_dims([SOS_tensor], 1), -1)

    for tx in range(args.max_length):
        dec_out, dec_hidden, attn_weights = decoder(dec_input, enc_out, dec_hidden)
        attn_weights = tf.reshape(attn_weights, (-1,))
        attention_plot[tx] = attn_weights.numpy()
        pred = tf.argmax(dec_out, axis=1).numpy()
        result += train.output_lang.index2word[pred[0]] + " "
        if train.output_lang.index2word[pred[0]] == "EOS":
            break
        dec_input = tf.expand_dims(pred, axis=1)
    return result, attention_plot


def main():
    main_parser = argparse.ArgumentParser(description="parser for translate module")
    main_parser.add_argument("--sentence", type=str,
                             help="input sentence to translate -- don't forget to enclosed input sentence in double quotation marks")
    main_parser.add_argument("--max_length", type=int, help="maximum length of output sentence ")
    args = main_parser.parse_args()
    translation, _ = translate(args)
    print("\n\n***input sentence: {}".format(args.sentence))
    print("\n***translation: {}".format(translation))


if __name__ == '__main__':
    main()
