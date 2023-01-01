import tensorflow as tf
from tensorflow import keras

from .attention import SelfAttentionBlock, CrossAttentionBlock, MaskedSelfAttentionBlock
from .ff_nn import FeedForward


class TransformerEncoderBlock(keras.layers.Layer):
    """An Encoder Layer/Block in Transformer architecture

    It takes data in shape (batch, seq, embedding) as input

    This can be stacked `N` times to be a part of Encoder-side

    Args:
        keras (_type_): _description_
    """

    def __init__(self,
                 embedding_dim: int,
                 num_heads: int,
                 dense_dim: int = 128,
                 dropout_rate: float = 0.1):
        super().__init__()

        self.self_attention = SelfAttentionBlock(num_heads=num_heads,
                                                 key_dim=embedding_dim,
                                                 dropout=dropout_rate)

        self.ff_nn = FeedForward(embedding_dim=embedding_dim, dense_dim=dense_dim)

        self.attn_scores = None

    def get_config(self):
        config = super().get_config()
        return config

    def call(self, x):
        x = self.self_attention(x)
        self.attn_scores = self.self_attention.attn_scores
        x = self.ff_nn(x)
        return x


class TransformerDecoderBlock(keras.layers.Layer):
    """A Decoder Layer/Block in Transformer architecture

    It takes `x, context` data in shape (batch, seq, embedding) as input
    where `context` are from the Encoder-side

    This can be stacked `N` times to be a part of Decoder-side

    Args:
        keras (_type_): _description_
    """

    def __init__(self,
                 embedding_dim: int,
                 num_heads: int,
                 dense_dim: int = 128,
                 dropout_rate: float = 0.1):
        super().__init__()

        self.masked_attention = MaskedSelfAttentionBlock(num_heads=num_heads,
                                                         key_dim=embedding_dim,
                                                         dropout=dropout_rate)

        self.cross_attention = CrossAttentionBlock(num_heads=num_heads,
                                                   key_dim=embedding_dim,
                                                   dropout=dropout_rate)

        self.ff_nn = FeedForward(embedding_dim=embedding_dim, dense_dim=dense_dim)

        self.attn_scores = None

    def get_config(self):
        config = super().get_config()
        return config

    def call(self, x, context):
        x = self.masked_attention(x=x)
        # query = x
        # key, value = context
        x = self.cross_attention(x=x, context=context)
        self.attn_scores = self.cross_attention.attn_scores
        x = self.ff_nn(x)
        return x