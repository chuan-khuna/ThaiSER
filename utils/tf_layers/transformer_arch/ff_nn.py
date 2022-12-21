import tensorflow as tf
from tensorflow import keras


class FeedForward(keras.layers.Layer):
    """A feed forward layer/block to add non-linearity with 2 layers
    with dropout and residual connection

    Dense(dense_dim)
    Dense(embedding_dim)

    Args:
        keras (_type_): _description_
    """

    def __init__(self, embedding_dim: int, dense_dim: int = 128, dropout_rate: float = 0.1):
        super().__init__()
        self.ff_nn = keras.models.Sequential([
            keras.layers.Dense(dense_dim),
            keras.layers.LeakyReLU(),
            keras.layers.Dense(embedding_dim),
            keras.layers.Dropout(dropout_rate)
        ])
        self.add = keras.layers.Add()
        self.layer_norm = keras.layers.LayerNormalization()

    def call(self, x):
        dense_out = self.ff_nn(x)
        x = self.add([x, dense_out])
        x = self.layer_norm(x)
        return x