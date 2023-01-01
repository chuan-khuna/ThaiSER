import tensorflow as tf
from tensorflow import keras

from .attention import *
from .transformer import TransformerDecoderBlock, TransformerEncoderBlock
from .embedding import PositionalEmbedding


class Encoder(keras.layers.Layer):

    def __init__(self,
                 *,
                 num_layers,
                 num_heads,
                 vocab_size,
                 embedding_dim,
                 max_positional_seq_length=2048,
                 dense_dim=128,
                 dropout_rate=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                                 embedding_dim=embedding_dim,
                                                 positional_seq_length=max_positional_seq_length)

        self.encoders = [
            TransformerEncoderBlock(embedding_dim=embedding_dim,
                                    num_heads=num_heads,
                                    dense_dim=dense_dim,
                                    dropout_rate=dropout_rate) for _ in range(self.num_layers)
        ]

        self.pos_enc_dropout = keras.layers.Dropout(dropout_rate)
        self.attn_scores = None

    def call(self, x):
        x = self.pos_embedding(x)
        x = self.pos_enc_dropout(x)

        for i in range(self.num_layers):
            x = self.encoders[i](x)

        self.attn_scores = self.encoders[-1].attn_scores

        return x


class Decoder(keras.layers.Layer):

    def __init__(self,
                 *,
                 num_layers,
                 num_heads,
                 vocab_size,
                 embedding_dim,
                 max_positional_seq_length=2048,
                 dense_dim=128,
                 dropout_rate=0.1):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                                 embedding_dim=embedding_dim,
                                                 positional_seq_length=max_positional_seq_length)

        self.decoders = [
            TransformerDecoderBlock(embedding_dim=embedding_dim,
                                    num_heads=num_heads,
                                    dense_dim=dense_dim,
                                    dropout_rate=dropout_rate) for _ in range(self.num_layers)
        ]

        self.pos_dec_dropout = keras.layers.Dropout(dropout_rate)
        self.attn_scores = None

    def call(self, x, context):
        x = self.pos_embedding(x)
        x = self.pos_dec_dropout(x)

        for i in range(self.num_layers):
            x = self.decoders[i](x, context)

        self.attn_scores = self.decoders[-1].attn_scores

        return x
