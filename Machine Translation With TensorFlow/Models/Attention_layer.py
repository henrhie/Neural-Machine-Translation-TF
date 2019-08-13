import tensorflow as tf

layers = tf.keras.layers


class AttentionLayer(tf.keras.layers.Layer):
    """
    This layer implements Bahdanau attention mechanism.
    Formula: score =  W1*(tanh(W2(query) + W3(values))
    attn_weights = softmax(score)
    context_vector = sum(attn_weights * values)
    """

    def __init__(self, num_units):
        super(AttentionLayer, self).__init__()

        self.fc1 = layers.Dense(num_units)
        self.fc2 = layers.Dense(num_units)
        self.fc3 = layers.Dense(1)

    def __call__(self, encoder_out, hidden):
        """
        encoder_out.shape = (batch_size, seq_length, hidden_size)
        hidden.shape = (batch_size, hidden_size)
        """
        score = self.fc3(tf.nn.tanh(self.fc1(encoder_out) +
                                    tf.expand_dims(self.fc2(hidden), axis=1)))
        # score.shape = (batch_size, seq_length, 1)

        attn_weights = tf.nn.softmax(score, axis=1)

        context_vector = attn_weights * encoder_out
        # context_vector.shape = (batch_size, seq_length, hidden_size)
        context_vector = tf.reduce_sum(context_vector, axis=1)
        # context_vector.shape = (batch_size, hidden_size)

        return context_vector, attn_weights
