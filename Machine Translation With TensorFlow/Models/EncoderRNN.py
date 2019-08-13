import tensorflow as tf

layer = tf.keras.layers


class EncoderRNN(tf.keras.models.Model):
    """
    implements the encoder part of the our sequence to sequence model
    consists of an embedding layer and GRU layer
    """

    def __init__(self, vocab_size, embedding_dim=256, batch_size=16, hidden_size=256):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embedding_layer = layer.Embedding(vocab_size, embedding_dim)
        self.GRU = layer.GRU(hidden_size, recurrent_initializer="glorot_uniform",
                             return_state=True, return_sequences=True)

    def call(self, inputs, hidden):
        embedded = self.embedding_layer(inputs)

        r_out, hidden = self.GRU(embedded, initial_state=hidden)
        """
        r_out.shape = (batch_size, seq_length, out_dim)
        hidden.shape = (batch_size, hidden_size)
        """
        return r_out, hidden

    def init_hidden(self):
        return tf.zeros(shape=(self.batch_size, self.hidden_size))
