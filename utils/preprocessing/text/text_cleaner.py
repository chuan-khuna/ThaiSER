import re
import string

################################
# regex note
################################
# \n, \r \t = white space
# \u200b-\u200c = zero-width white space
# urls
# https://regex101.com/r/hG9t0Q/1
# https://regexr.com/3e6m0
# https://regexr.com/37i6s
# ref pythainlp.utils.normalize


class TextCleaner:
    """
    TextCleaner is a set of regular expressions patterns
    that used for cleaning texts before tokenising them.

    Included regular expression for white spaces, urls, punctuations

    """

    def __init__(self, remove_punctuations: bool = True):
        self.remove_punctuations = remove_punctuations

    def _replace_special_white_spaces(self, text: str) -> str:
        pattern = re.compile(r"[\n|\r|\t|\u200b-\u200c]")
        return re.sub(pattern, " ", text)

    def _replace_duplicated_spaces(self, text: str) -> str:
        pattern = re.compile(r"(?:\s)(\s+)")
        return re.sub(pattern, " ", text)

    def _replace_start_end_spaces(self, text: str) -> str:
        pattern = re.compile(r"^\s*")
        text = re.sub(pattern, "", text)
        pattern = re.compile(r"\s*$")
        text = re.sub(pattern, "", text)
        return text

    def _remove_urls(self, text: str) -> str:
        pattern = re.compile(
            r"(http(s)?:\/\/.)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{0,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)"
        )
        # pattern = re.compile(r"https?:\/\/.*?[\s+]")
        return re.sub(pattern, "", text)

    def _remove_punctuations(self, text: str) -> str:
        return text.translate(str.maketrans("", "", string.punctuation))

    def clean(self, text: str) -> str:
        text = self._replace_special_white_spaces(text)
        text = self._remove_urls(text)
        if self.remove_punctuations:
            text = self._remove_punctuations(text)
        # to ensure that there is no starting and ending space
        text = self._replace_duplicated_spaces(text)
        text = self._replace_start_end_spaces(text)
        return text
