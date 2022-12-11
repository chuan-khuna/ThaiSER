import gensim
import re


class NGram:
    """An N-Gram helper based on gensim.models.phrases.Phrases
    https://radimrehurek.com/gensim/models/phrases.html#
    """

    def __init__(self, max_n: int = 5):
        self.models = {}
        self.max_n = max_n
        self.delimiter = '_'
        self.only_ngram = False
        self.remove_delim = True
        self.stopwords: list[str] = []

    def fit(self, tokenised_texts: list[list[str]], **kwargs) -> dict[int, gensim.models.Phrases]:
        """train 2 to max_n-gram models
        models are stored in dictionary

        Args:
            tokenised_texts (list[list[str]]): list of tokenised texts(ie list of words)

        Returns:
            dict[int, gensim.models.Phrases]: dictionary of n: n-gram model
        """

        # initialise 1-gram text
        prev_gram_texts = tokenised_texts

        for n in range(2, self.max_n + 1):
            model = gensim.models.Phrases(prev_gram_texts, delimiter=self.delimiter, **kwargs)
            self.models[n] = model
            prev_gram_texts = [model[text] for text in prev_gram_texts]

    def _remove_delimiter(self, n_gram: list[str]) -> list[str]:
        if self.remove_delim:
            n_gram = [re.sub(self.delimiter, "", pair) for pair in n_gram]
        return n_gram

    def _filter_n_gram(self, n_gram: list[str], n: int) -> list[str]:
        if self.only_ngram:
            new_ngram = []
            for pair in n_gram:
                if len(pair.split(self.delimiter)) == n:
                    new_ngram.append(pair)
            n_gram = new_ngram
        return n_gram

    def _filter_stopwords(self, n_gram: list[str]) -> list[str]:
        non_stopwords_ngram = []
        for ngram_word in n_gram:
            all_stopword = True
            for token in ngram_word.split(self.delimiter):
                all_stopword = all_stopword and (token in self.stopwords)
            if not all_stopword:
                non_stopwords_ngram.append(ngram_word)
        return non_stopwords_ngram

    def get_ngrams(self, tokenised_text: list[str]) -> dict[int, list[str]]:
        """get 2-gram(bigram) to max_n-gram
        set only_ngram=True to filter only n-gram, otherwise it will return all words with n-gram pairs

        set remove_delim=True to replace relimiter

        Args:
            tokenised_text (list[str]): 1-gram tokenised text, ie a list of words

        Returns:
            dict[int, list[str]]: dictionary of {n: n-gram pairs}
        """
        n_grams = {}

        prev_gram = tokenised_text
        for n, model in self.models.items():
            n_gram = model[prev_gram]
            prev_gram = n_gram
            n_gram = self._filter_stopwords(n_gram)
            n_gram = self._filter_n_gram(n_gram, n)
            n_gram = self._remove_delimiter(n_gram)
            n_grams[n] = n_gram

        return n_grams
