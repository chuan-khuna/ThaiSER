import numpy as np
import tensorflow as tf
import tensorflow_io as tfio


class AudioPreprocessor:
    """
    A preprocessor utils for 1 audio sample

    - This will process only audio data, not include labels
    - This will prepare audio data for training and testing, demoing
    - for training process you should call this in your own wrapper
    """

    def __init__(self):

        # audio to spectrogram setting
        # window length must <= nfft
        self.nfft = 512
        self.window_length = 512
        # aka hop length
        self.stride = 256

        # spectrogram to melspectrogram setting
        self.mel_sr = 44100
        self.n_mels = 128
        self.max_db = 80

        # setting for CNN layer
        self.img_h = 128
        self.img_w = 1024
        self.img_ch = 3

    def from_file(self, filename: str) -> tf.Tensor:
        """Use tensorflow utils to read audio file(.flac)
        """
        file_content = tf.io.read_file(filename)
        audio_tensor = tfio.audio.decode_flac(file_content, dtype=tf.int16)
        audio_tensor = audio_tensor[:, 0]
        return audio_tensor

    def from_array(self, array: np.ndarray):
        """get 1-d numpy array and convert into 1-d tensor
        """
        return tf.constant(array)

    def _normalize(self, audio_tensor: tf.Tensor) -> tf.Tensor:
        # https://www.tensorflow.org/io/tutorials/audio
        return tf.cast(audio_tensor, tf.float32) / tf.constant(32768.0)

    def _to_spectrogram(self, audio_tensor: tf.Tensor) -> tf.Tensor:
        spectogram_tensor = tfio.audio.spectrogram(audio_tensor,
                                                   nfft=self.nfft,
                                                   window=self.window_length,
                                                   stride=self.stride)
        return spectogram_tensor

    def _spectrogram_to_melspectrogram(self, spectogram_tensor: tf.Tensor) -> tf.Tensor:
        melspec_tensor = tfio.audio.melscale(spectogram_tensor,
                                             rate=self.mel_sr,
                                             mels=self.n_mels,
                                             fmin=0,
                                             fmax=int(self.mel_sr // 2))
        return melspec_tensor

    def _melspec_to_db_mel_spec(self, melspec_tensor: tf.Tensor) -> tf.Tensor:
        return tfio.audio.dbscale(melspec_tensor, top_db=self.max_db)

    def _expand_dims(self, tensor: tf.Tensor) -> tf.Tensor:
        return tf.repeat(tf.expand_dims(tensor, -1), self.img_ch, -1)

    def _swap_axis(self, tensor: tf.Tensor) -> tf.Tensor:
        return tf.transpose(tensor, perm=[1, 0, 2])

    def _resize_img(self, tensor: tf.Tensor) -> tf.Tensor:
        return tf.image.resize(tensor, size=(self.img_h, self.img_w))

    def _to_timestep_first(self, tensor: tf.Tensor) -> tf.Tensor:
        """Convert an image-like spectrogram of (h, w, ch) = (freq, time, ch)
        to an RNN-input, ie (time, freq)
        """
        tensor = self._swap_axis(tensor)
        return tf.squeeze(tensor)

    def preprocess(self, audio_tensor: tf.Tensor) -> tf.Tensor:
        """preprocess audio data
        from 1-d audio tensor to db melspectrogram image

        Args:
            audio_tensor (tf.Tensor): 1-d audio tensor in shape (n, ) and type int16
            eg [1, 2, 3, ...]

        Returns:
            tf.Tensor: a melspectrogram tensor in db; compatible with CNN - (h, w, ch) = (feature, timestep, channel)
        """
        # preprocessing audio file
        tensor = self._normalize(audio_tensor)
        tensor = self._to_spectrogram(tensor)
        tensor = self._spectrogram_to_melspectrogram(tensor)
        tensor = self._melspec_to_db_mel_spec(tensor)

        # convert a melspectrogram array to an image array for CNN
        tensor = self._expand_dims(tensor)
        tensor = self._swap_axis(tensor)
        tensor = self._resize_img(tensor)
        return tensor