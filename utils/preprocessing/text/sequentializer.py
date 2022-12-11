OOV_TOKEN = "<OOV>"


class Sequentializer:
    def __init__(self, oov_token: str | None = OOV_TOKEN, drop_stopwords: bool = True):
        self.oov_token = oov_token
        self.drop_stopwords = drop_stopwords

        # vocab and inverse vocab should be synced
        # so I will set them as protected variables
        self._vocab = None
        self._inverse_vocab = None
        self.stopwords: list[str] | None = None

    @property
    def vocab(self):
        return self._vocab

    @property
    def inverse_vocab(self):
        return self._inverse_vocab

    @vocab.setter
    def vocab(self, vocab: dict):
        if vocab is None:
            self._vocab = None
            self._inverse_vocab = None
        else:
            self._vocab = vocab
            self._inverse_vocab = {}
            for key, val in self.vocab.items():
                self._inverse_vocab[val] = key

    def remove_stopwords(self, text: list[str]) -> list[str]:
        if self.drop_stopwords and self.stopwords:
            text_without_stopwords = []
            for word in text:
                if word not in self.stopwords:
                    text_without_stopwords.append(word)
            return text_without_stopwords
        else:
            return text

    def _get_oov_token_from_dict(self, dict_: dict) -> str | int | None:
        if self.oov_token in dict_.keys():
            return dict_[self.oov_token]
        else:
            return None

    def _map_keys_to_values(
        self, keys_: list[str | int], dict_: dict
    ) -> list[str | int]:
        sequence = []
        for k in keys_:
            if k in dict_.keys():
                sequence.append(dict_[k])
            elif oov := self._get_oov_token_from_dict(dict_):
                sequence.append(oov)
        return sequence

    def text_to_sequence(self, text: list[str]) -> list[int]:
        if self._vocab is None:
            raise ValueError(
                "Vocabulary not found, set if before converting text to sequence"
            )
        return self._map_keys_to_values(text, self.vocab)

    def sequence_to_text(self, sequence: list[int]) -> list[str]:
        if self._vocab is None:
            raise ValueError(
                "Vocabulary not found, set if before converting text to sequence"
            )
        return self._map_keys_to_values(sequence, self.inverse_vocab)

    # easter egg
    sequentialize = sequentialise = text_to_sequence
    desequentialize = desequentialise = sequence_to_text


# Just for my British English Style
Sequentialiser = Sequentializer
