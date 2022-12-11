import pythainlp


class CustomTokenizer:

    def __init__(self, custom_words: list[str], engines: list[str] = ('newmm', 'newmm')):
        """initialise the tokeniser

        Args:
            custom_words (list[str]): list of custom words
            engines (list[str], optional): _description_. Defaults to ('newmm', 'newmm').
        """
        self.custom_words = custom_words
        self.custom_dict = pythainlp.util.trie.Trie(custom_words)
        self.engines = engines
        self.keep_whitespace = False

    def tokenize(self, text: str):
        """Tokenise text with a customer tokeniser first
        Then tokenise with the default tokeniser

        Args:
            text (str): text to tokenise

        Returns:
            list to tokens: list[str]
        """
        # tokenise with custom dict
        tokenised_text = pythainlp.tokenize.word_tokenize(text,
                                                          custom_dict=self.custom_dict,
                                                          keep_whitespace=self.keep_whitespace,
                                                          engine=self.engines[0])

        tokens = []
        # tokenise with default tokeniser
        for token in tokenised_text:
            if token in self.custom_dict.words:
                tokens.append(token)
            else:
                default_tokenised_text = pythainlp.tokenize.word_tokenize(
                    token, keep_whitespace=self.keep_whitespace, engine=self.engines[1])
                tokens += default_tokenised_text
        return tokens
