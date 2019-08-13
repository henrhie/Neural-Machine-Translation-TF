import tensorflow as tf
from Models.Attention_layer import AttentionLayer

layer = tf.keras.layers


class DecoderRNN(tf.keras.models.Model):
    """
    implementation of the decoder part of our sequence to sequence model
    consist of an attention layer, GRU layer and a fully connected layer
    """

    def __init__(self, out_vocab_size, num_units=256, embedding_dim=256):
        super(DecoderRNN, self).__init__()

        self.attn_layer = AttentionLayer(num_units)
        self.GRU = layer.GRU(num_units, recurrent_initializer="glorot_uniform",
                             return_sequences=True, return_state=True)
        self.embedding = layer.Embedding(out_vocab_size, embedding_dim)
        self.fc = layer.Dense(out_vocab_size)

    def call(self, x, encoder_out, encoder_hidden):
        context_vector, attn_weights = self.attn_layer(encoder_out, encoder_hidden)
        x = self.embedding(x)
        """"
        x.shape = (batch_size, seq_length, embedding_dim) // (batch_size, 1, embedding_dim
        context_vector.shape = (batch_size, hidden_size)
        """
        context_vector = tf.expand_dims(context_vector, axis=1)  # shape = (batch_size, 1, hidden_size)
        inp = tf.concat([context_vector, x], axis=-1)  # shape = (batch_size, 1, hidden_size + embedding_dim)
        r_out, dec_hidden = self.GRU(inp, initial_state=encoder_hidden)  # r_out.shape = (batch_size, 1, num_units)

        out = tf.reshape(r_out, shape=(-1, r_out.shape[2]))
        out = self.fc(out)
        return out, dec_hidden, attn_weights

