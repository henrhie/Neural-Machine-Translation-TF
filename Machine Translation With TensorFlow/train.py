import tensorflow as tf
import numpy as np
from Models.EncoderRNN import EncoderRNN
from Models.DecoderRNN import DecoderRNN
import Utils
import argparse
import os

data_path = "data/fra.txt"
SOS_token = 0
EOS_token = 1
optimizer = tf.keras.optimizers.Adam()
input_tensor, target_tensor, input_lang, output_lang, pairs = Utils.build_lang("fr", "eng", data_path)
Encoder = EncoderRNN(input_lang.n_words, 256, batch_size=1)
Decoder = DecoderRNN(output_lang.n_words)


def train_step(encoder, decoder, inputs, targets, enc_hidden):
    loss = 0.0
    with tf.GradientTape() as tape:
        batch_size = inputs.shape[0]
        enc_output, enc_hidden = encoder(inputs, enc_hidden)

        SOS_tensor = np.array([SOS_token])
        dec_input = tf.squeeze(tf.expand_dims([SOS_tensor] * batch_size, 1), -1)
        dec_hidden = enc_hidden

        for tx in range(targets.shape[1] - 1):
            dec_out, dec_hidden, _ = decoder(dec_input, enc_output, dec_hidden)
            loss += Utils.loss_fn(targets[:, tx], dec_out)
            dec_input = tf.expand_dims(targets[:, tx], 1)

    batch_loss = loss / targets.shape[1]
    t_variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, t_variables)
    optimizer.apply_gradients(zip(gradients, t_variables))
    return batch_loss


def train_loop(encoder, decoder, _dataset, _steps_per_epoch, args):
    if os.path.exists("checkpoint/encoder.h5") and \
            os.path.exists("checkpoint/decoder.h5"):
        print("loading models for training.....")
        hidden = [tf.zeros((1, 256))]
        _ = train_step(encoder, decoder, tf.ones((1, 2)),
                       tf.ones((1, 2)), hidden)
        encoder.load_weights("checkpoint/encoder.h5")
        decoder.load_weights("checkpoint/decoder.h5")
        print("models loaded for training :)")
    for e in range(1, args.epochs):

        total_loss = 0.0
        hidden = encoder.init_hidden()

        for idx, (inps, target) in enumerate(_dataset.take(_steps_per_epoch)):

            batch_loss = train_step(encoder, decoder, inps, target, hidden)
            total_loss += batch_loss

            if idx % args.log_every == 0:
                print("Epochs: {} batch_loss: {:.4f}".format(e, batch_loss))
                Utils.checkpoint(encoder, 'encoder')
                Utils.checkpoint(decoder, 'decoder')

        if e % 2 == 0:
            print("Epochs: {}/{} total_loss: {:.4f}".format(
                e, args.epochs, total_loss / _steps_per_epoch))


def main():
    main_parser = argparse.ArgumentParser(description="training argument parser")
    main_parser.add_argument("--epochs", type=int, default=10,
                             help="number of epochs")
    main_parser.add_argument("--batch_size", type=int, default=16,
                             help="batch size of dataset")
    main_parser.add_argument("--log_every", type=int, default=10,
                             help="stages to log statistics and save model")
    main_parser.add_argument("--buffer_size", type=int, default=10000, help="buffer size for dataset")

    args = main_parser.parse_args()
    Encoder = EncoderRNN(input_lang.n_words, 256, batch_size=args.batch_size)
    Decoder = DecoderRNN(output_lang.n_words)

    print("TensorFlow version: {}".format(tf.__version__))

    steps_per_epoch = len(pairs) // args.batch_size
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(args.buffer_size)
    dataset = dataset.batch(args.batch_size)
    train_loop(Encoder, Decoder, dataset, steps_per_epoch, args)


if __name__ == '__main__':
    main()
