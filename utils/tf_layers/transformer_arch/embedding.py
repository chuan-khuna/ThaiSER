# ref: https://machinelearningmastery.com/the-transformer-positional-encoding-layer-in-keras-part-2/
# https://www.tensorflow.org/text/tutorials/transformer#the_embedding_and_positional_encoding_layer

# don't forget to add mask_zero at the Embedding layer
# preprocessing token to vector - https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization
# Embedding masking - https://www.tensorflow.org/text/tutorials/transformer#the_embedding_and_positional_encoding_layer
# evaluation metrics - https://www.tensorflow.org/text/tutorials/transformer#set_up_the_loss_and_metrics
# RNN + Embedding Use masking to handle the variable sequence lengths -  https://www.tensorflow.org/text/tutorials/text_classification_rnn
# https://www.tensorflow.org/guide/keras/masking_and_padding

import tensorflow as tf
from tensorflow import keras
import numpy as np


def get_positional_encoding_values(seq_length: int, embed_dim: int, n: int = 10000) -> np.array:

    # PE(pos, 2i) = sin( pos/(10000**(2i/dim)) )
    # PE(pos, 2i+1) = cos( pos/(10000**(2i/dim)) )
    # i = i-th dimension of embedding vector

    # i
    dimensions = np.arange(embed_dim)
    # masking odd, even i; 2i
    even_mask = dimensions % 2 == 0
    odd_mask = dimensions % 2 == 1
    # shape (1, dim)
    dimensions = dimensions[np.newaxis, :]

    # pos
    positions = np.arange(seq_length)
    # shape (pos, 1)
    positions = positions[:, np.newaxis]

    dimensions = dimensions / embed_dim
    rate = 1 / (n**(dimensions))
    # shape (pos, dim)
    rads = positions * rate

    positional_encoding = np.sin(rads) * even_mask + np.cos(rads) * odd_mask

    assert positional_encoding.shape == (seq_length, embed_dim)

    # should cast data type to tensorflow's
    # otherwise it will error when run model.fit()
    # TODO: how to make this code compatible to mixed precision policy
    return tf.constant(positional_encoding, dtype=tf.float32)


class FixedPositionalEncoding(keras.layers.Layer):
    """Add "Positional Encodings" to embedding vectors

    How to use:

    model.add(Embedding(..., mask_zero=True))
    model.add(FixedPositionalEncoding(....))

    Ref: Attention is All You Need (Vaswani et al)

    Args:
        keras (_type_): _description_
    """

    def __init__(self, seq_length: int, embed_dim: int, n: int = 10000):
        super().__init__()

        self.embed_dim = embed_dim
        self.positional_encoding = get_positional_encoding_values(seq_length, self.embed_dim, n)

        self.add = keras.layers.Add()

    def get_config(self):
        config = super().get_config()
        return config

    def call(self, x):
        # input shape: (batch, seq, emb)
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[1]

        # we can define a very long positional encoding, say 2048
        # then access only sequence length
        pos_encoding = self.positional_encoding
        pos_encoding = pos_encoding[tf.newaxis, :seq_length, :]

        # scaling x, before adding positional encoding
        x = x * tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))

        # TESTED
        # we can also use Add() layer by
        # repeat constant in the same shape of data
        # pos_encoding = tf.repeat(pos_encoding, batch_size, axis=0)
        # x = self.add([x, pos_encoding])

        # TESTED
        # with + operator, no need to use tf.repeat
        x = x + pos_encoding
        return x


class PositionalEmbedding(keras.layers.Layer):
    # Embedding with fixed positional encoding in one block
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 positional_seq_length: int = 2048,
                 n: int = 10000):
        """_summary_

        Args:
            vocab_size (int): _description_
            embedding_dim (int): _description_
            n (int, optional): _description_. Defaults to 10000.
        """
        super().__init__()
        self.embed_dim = embedding_dim

        self.embedding = keras.layers.Embedding(input_dim=vocab_size,
                                                output_dim=self.embed_dim,
                                                mask_zero=True)

        self.positional_encoding = get_positional_encoding_values(positional_seq_length,
                                                                  self.embed_dim, n)

    def get_config(self):
        config = super().get_config()
        return config

    # why?
    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[1]

        x = self.embedding(x)
        # scaling x, before adding positional encoding
        x = x * tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))

        pos_encoding = self.positional_encoding
        pos_encoding = pos_encoding[tf.newaxis, :seq_length, :]
        x = x + pos_encoding

        return x