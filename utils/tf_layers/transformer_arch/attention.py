import tensorflow as tf
from tensorflow import keras


class BaseMultiHeadedAttentionBlock(keras.layers.Layer):
    """An Attention Block consists operations:
    - perform attention mechanism
    - residual connection x, attention(x)

    This block takes data in shape (batch, seq, embedding) as input

    Args:
        keras (_type_): _description_
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.attention = keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = keras.layers.LayerNormalization()
        self.residual_add = keras.layers.Add()

    def get_config(self):
        config = super().get_config()
        return config


class SelfAttentionBlock(BaseMultiHeadedAttentionBlock):

    def call(self, x):
        attn_output, self.attn_scores = self.attention(query=x,
                                                       key=x,
                                                       value=x,
                                                       return_attention_scores=True)
        x = self.residual_add([x, attn_output])
        x = self.layernorm(x)

        # shape: batch_size, seq_length, embedding_dim
        return x


class MaskedSelfAttentionBlock(BaseMultiHeadedAttentionBlock):

    def call(self, x):
        attn_output = self.attention(query=x, key=x, value=x, use_causal_mask=True)
        x = self.residual_add([x, attn_output])
        x = self.layernorm(x)

        # shape: batch_size, seq_length, embedding_dim
        return x


class CrossAttentionBlock(BaseMultiHeadedAttentionBlock):

    def call(self, x, context):
        attn_output, self.attn_scores = self.attention(query=x,
                                                       key=context,
                                                       value=context,
                                                       return_attention_scores=True)
        x = self.residual_add([x, attn_output])
        x = self.layernorm(x)

        # shape: batch_size, context_length, embedding_dim
        return x