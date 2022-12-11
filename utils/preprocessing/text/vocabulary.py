import pandas as pd

OOV_TOKEN = "<OOV>"
SPECIAL_TOKENS = [OOV_TOKEN]


class Vocabulary:
    def __init__(self, special_tokens: list[str] = SPECIAL_TOKENS):
        self._special_tokens = special_tokens

        # protected attributes
        self._vocab = {}
        self._inverse_vocab = {}
        self._words = []
        self._df = None

        # initilise all attributes
        self.from_list([])

    @property
    def vocab(self):
        return self._vocab

    @property
    def inverse_vocab(self):
        return self._inverse_vocab

    @property
    def words(self):
        return self._words

    @property
    def df(self):
        return self._df

    @property
    def special_tokens(self):
        return self._special_tokens

    @special_tokens.setter
    def special_tokens(self, special_tokens: None | list[str]):
        # backup old tokens
        prev_special_tokens = self.special_tokens.copy()

        # set new tokens
        # to ensure that the variable is in list format
        if special_tokens is None:
            self._special_tokens = []
        elif isinstance(special_tokens, list):
            self._special_tokens = special_tokens
        else:
            raise TypeError(
                "Special tokens should be a list of strings or None, [] empty list"
            )

        # if words is not None -> words are imported/initialised
        # this function should update new vocabulary
        if prev_special_tokens is None:
            special_tokens_size = 0
        else:
            special_tokens_size = len(prev_special_tokens)

        words_without_tokens = self.words[special_tokens_size:]
        self.from_list(words_without_tokens)

    def _reset_attributes(self):
        self._vocab = {}
        self._inverse_vocab = {}
        self._words = None
        self._df = None

    def from_list(self, words: list[str]):
        self._reset_attributes()
        for i, token in enumerate(self.special_tokens + words):
            if token not in self.vocab.keys():
                self._vocab[token] = i + 1
                self._inverse_vocab[i + 1] = token
        self._words = list(self.vocab.keys())

    def from_df(self, df: pd.DataFrame, col: str):
        self._reset_attributes()
        self.from_list(df[col].tolist())
        self._df = df
