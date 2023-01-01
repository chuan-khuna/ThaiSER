import json
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np


class AudioPreprocessor:

    def __init__(self):
        self.sampling_rate = 16000

        # spectrorgram setting
        self.nfft = 512
        self.window = 512
        self.stride = 256

        # spectrogram to melspectrogram setting
        self.n_mels = 128
        self.max_db = 80

        # padding/truncating (sec)
        self.pad_duration = 30

    def from_json(self, file: str):
        with open(file, "r") as f:
            setting = json.load(f)

        for k in self.__dict__.keys():
            self.__dict__[k] = setting[k]

    def save_setting(self, file: str):
        with open(file, "w") as f:
            f.write(json.dumps(self.__dict__, indent=2))

    def _resampling(self, audio_tensor: tf.Tensor, rate: int) -> tf.Tensor:
        """resampling audio tensor from the original sampling rate `rate` to `self.sampling_rate`

        Args:
            audio_tensor (tf.Tensor): audio tensor in shape `(length, )`
            rate (int): sampling rate of the original audio

        Returns:
            tf.Tensor: resampled audio tensor in shape `(length, )`
        """
        if (self.sampling_rate != rate) and (self.sampling_rate is not None):
            audio_tensor = tfio.audio.resample(audio_tensor, rate, self.sampling_rate)
        return audio_tensor

    def _normalise(self, audio_tensor: tf.Tensor) -> tf.Tensor:
        """Normalise audio tensor by dividing the max value of int16 `32768.0`

        Args:
            audio_tensor (tf.Tensor): audio tensor in shape `(length, )`

        Returns:
            tf.Tensor: audio tensor in shape `(length, )`
        """
        # https://www.tensorflow.org/io/tutorials/audio
        # https://keras.io/examples/audio/transformer_asr/#preprocess-the-dataset
        audio_tensor = tf.cast(audio_tensor, tf.float32) / tf.cast(tf.int16.max + 1,
                                                                   dtype=tf.float32)
        return audio_tensor

    def _pad_signal(self, audio_tensor: tf.Tensor) -> tf.Tensor:
        """pad/trim wave tensor to shape `pad_duration(sec) * sampling_rate`

        Args:
            audio_tensor (tf.Tensor): _description_

        Returns:
            tf.Tensor: audio tensor in shape (pad_duration(sec) * sampling_rate, )
        """
        # https://keras.io/examples/audio/transformer_asr/#preprocess-the-dataset

        # deprecated
        # audio_tensor = tf.keras.utils.pad_sequences(audio_tensor[tf.newaxis, :],
        #                                             maxlen=self.pad_duration * self.sampling_rate,
        #                                             dtype='float32',
        #                                             padding='post')

        max_len = self.pad_duration * self.sampling_rate
        audio_len = tf.shape(audio_tensor)[0]

        if audio_len > max_len:
            pad_len = 0
        else:
            pad_len = max_len - audio_len

        paddings = [[0, 0], [0, pad_len]]
        audio_tensor = tf.pad(audio_tensor[tf.newaxis], paddings)
        audio_tensor = tf.squeeze(audio_tensor)[:max_len]
        return audio_tensor

    def _to_spectrogram(self, audio_tensor: tf.Tensor) -> tf.Tensor:
        """Convert audio wave tensor to spectrogram tensor

        Args:
            audio_tensor (tf.Tensor): audio tensor in shape `(length, )`

        Returns:
            tf.Tensor: spectrogram tensor in shape `(spec length, features)
        """
        spectogram_tensor = tfio.audio.spectrogram(audio_tensor,
                                                   nfft=self.nfft,
                                                   window=self.window,
                                                   stride=self.stride)
        return spectogram_tensor

    def _spectrogram_to_melspectrogram(self, spectogram_tensor: tf.Tensor) -> tf.Tensor:
        """convert spectrogram to mel-spectrogram

        Args:
            spectogram_tensor (tf.Tensor): tensor in shape `(spec length, features)`

        Returns:
            tf.Tensor: tensor in shape `(spec length, features)`
        """
        melspec_tensor = tfio.audio.melscale(spectogram_tensor,
                                             rate=self.sampling_rate,
                                             mels=self.n_mels,
                                             fmin=0,
                                             fmax=int(self.sampling_rate // 2))
        return melspec_tensor

    def _melspec_to_db_melspec(self, melspec_tensor: tf.Tensor) -> tf.Tensor:
        return tfio.audio.dbscale(melspec_tensor, top_db=self.max_db)

    def _process_audio_tensor_to_melspectrogram(self, audio_tensor: tf.Tensor) -> tf.Tensor:
        """A wrapper function to process audio wave to mel-spectrogram(db scale)

        Args:
            audio_tensor (tf.Tensor): audio tensor in shape `(length, )`

        Returns:
            tf.Tensor: spectrogram tensor in shape `(spec length, features)
        """
        spec = self._to_spectrogram(audio_tensor)
        mel_spec = self._spectrogram_to_melspectrogram(spec)
        mel_spec_db = self._melspec_to_db_melspec(mel_spec)
        return mel_spec_db

    def _audio_tensor_from_file(self, filename: str) -> tf.Tensor:
        """Load audio wave from `.flac` file

        Args:
            filename (str): _description_

        Returns:
            tf.Tensor: tensor in shape `(length, )`
        """

        content = tfio.IOTensor.graph(tf.int16).from_audio(filename)
        rate = tf.cast(content.rate, dtype=tf.int64)
        audio_tensor = content.to_tensor()
        audio_tensor = tf.squeeze(audio_tensor[:, 0])
        return audio_tensor, rate

    def preprocess(self, filename: str) -> tf.Tensor:
        """Preprocess an audio file

        Args:
            filename (str): _description_

        Returns:
            tf.Tensor: 2 tensors, audio wave and mel-spectrogram
        """

        audio_tensor, rate = self._audio_tensor_from_file(filename)
        audio_tensor = self._resampling(audio_tensor, rate)
        audio_tensor = self._normalise(audio_tensor)
        audio_tensor = self._pad_signal(audio_tensor)
        mel_spec = self._process_audio_tensor_to_melspectrogram(audio_tensor)
        return audio_tensor, mel_spec