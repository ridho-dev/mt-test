import tensorflow as tf

from keras.layers import Dense, Embedding, GRU

class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, enc_units, **kwargs):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.enc_units = enc_units
        self.embedding = Embedding(self.vocab_size, self.embedding_dim, name="embedding_layer_encoder")
        self.gru = GRU(self.enc_units, return_state=True, return_sequences=True, name="Encoder_GRU")

    def call(self, inputs, hidden, **kwargs):
        input_embeds = self.embedding(inputs)
        gru_output, gru_state_h = self.gru(input_embeds, initial_state=hidden)
        return gru_output, gru_state_h

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({"vocab_size": self.vocab_size,
                       "embedding_dim": self.embedding_dim,
                       "enc_units": self.enc_units})
        return config

    def initialize_hidden_state(self, batch_sz):
        return tf.zeros((batch_sz, self.enc_units))


class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dec_units = dec_units
        self.dense = Dense(vocab_size)
        self.W1 = Dense(self.dec_units)
        self.W2 = Dense(self.dec_units)
        self.V = Dense(1)
        self.embedding = Embedding(self.vocab_size, self.embedding_dim, name="embedding_layer_decoder")
        self.gru = GRU(self.dec_units, return_state=True, return_sequences=True,
                       recurrent_initializer='glorot_uniform', name="Encoder_GRU")

    def call(self, target_sentences, enc_output, state_h):
        # target_embeds dimension = (None, None, 256)
        target_embeds = self.embedding(target_sentences)
        # expand layer dimensions to fit encoder output tensor, state_h(1, 1024) --> hidden_with_time_axis (1, 1, 1024)
        hidden_with_time_axis = tf.expand_dims(state_h, 1)

        # ATTENTION MECHANISM
        # calculate alignment score, shape alignment score (1, 18, 1)
        score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis)))
        # shape attention_weight (1, 18, 1) e(i)/sum(e(i)+...+e(n))
        attention_weights = tf.nn.softmax(score, axis=1)
        # context vector before reduce_sum (1, 18, 1024) = attention_weight (1, 18, 1) * encoder output (1, 18, 1024)
        context_vector = attention_weights * enc_output
        # set up the context vector to match the embeds target,
        # the context vector after reduce_sum (1, 1024) sum 18 columns
        context_vector = tf.reduce_sum(context_vector, axis=1)
        # context vector expand_dims (1, 1, 1024)
        # concatenation context vector and target embeddings # target embed after concat(1, 1, 1280)
        target_embeds = tf.concat([tf.expand_dims(context_vector, 1), target_embeds], axis=-1)

        # GATED RECURRENT UNITS
        # receive result from gru ,gru output before reshape (1, 1, 1024)
        gru_output, state_h = self.gru(target_embeds)
        # Make the output according to the dense layer for (vocab representation),
        # gru output after reshape (1, 1024) gru_output.shape[2] = 1024
        gru_output = tf.reshape(gru_output, (-1, gru_output.shape[2]))
        # get output -> size of vocab , dec_output(1, 2354)
        dec_output = self.dense(gru_output)
        return attention_weights, dec_output, state_h, attention_weights

    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({"vocab_size": self.vocab_size,
                       "embedding_dim": self.embedding_dim,
                       "dec_units": self.dec_units,
                       "dense": self.dense,
                       "W1": self.W1,
                       "W2": self.W2,
                       "V": self.V})
        return config

    def initialize_hidden_state(self, batch_sz):
        return tf.zeros((batch_sz, self.enc_units))
