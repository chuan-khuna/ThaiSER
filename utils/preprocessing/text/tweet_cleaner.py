import re
from .text_cleaner import TextCleaner


class TweetCleaner(TextCleaner):

    # __init__ method from text cleaner
    def __init__(self, remove_punctuations: str = True):
        TextCleaner.__init__(self, remove_punctuations=remove_punctuations)
        self.user_pattern = r"\@[a-zA-Z0-9_+=]{3,}"
        self.RT_user_pattern = r"^RT\s(\@[a-zA-Z0-9_+=]{3,}):\s"
        self.RT_pattern = r"^RT\s\@[a-zA-Z0-9_+=]{3,}:\s"

    def get_RT_user(self, text: str) -> str:
        pattern = re.compile(self.RT_user_pattern)
        RT_user = re.findall(pattern, text)
        if RT_user:
            return RT_user[0]
        else:
            return None

    def get_mentioned_users(self, text: str) -> str:
        pattern = re.compile(self.user_pattern)
        mentioned_users = re.findall(pattern, text)
        if mentioned_users:
            return mentioned_users
        else:
            return None

    def _replace_RT_start(self, text: str) -> str:
        pattern = re.compile(self.RT_pattern)
        return re.sub(pattern, "", text)

    def _replace_all_users(self, text: str) -> str:
        pattern = re.compile(self.user_pattern)
        return re.sub(pattern, "", text)

    def clean(self, text: str) -> str:
        # TweetCleaner is inherited from TextCleaner
        # Call parent methods
        text = self._replace_special_white_spaces(text)
        text = self._remove_urls(text)
        text = self._replace_RT_start(text)
        text = self._replace_all_users(text)
        text = super().clean(text)
        return text
