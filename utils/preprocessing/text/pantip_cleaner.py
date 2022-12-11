from .text_cleaner import TextCleaner
import re
import html


class PantipCleaner(TextCleaner):

    def __init__(self, remove_punctuations: bool = True):
        TextCleaner.__init__(self, remove_punctuations=remove_punctuations)

        self.pantip_spoil_pattern = r"\[Spoil\] คลิกเพื่อดูข้อความที่ซ่อนไว้"

    def _replace_spoil_component(self, text: str) -> str:
        pattern = re.compile(self.pantip_spoil_pattern)
        return re.sub(pattern, "", text)

    def _replace_edit_text(self, text: str) -> str:
        pattern = re.compile(r"แก้ไขข้อความเมื่อ")
        return re.sub(pattern, "", text)

    def _replace_pantip_spaces(self, text: str) -> str:
        space_patterns = [r"\{\{eem\}\}", r"\{\{em\}\}"]
        for pattern in space_patterns:
            pattern = re.compile(pattern)
            text = re.sub(pattern, "", text)
        return text

    def _unescape_html(self, text: str) -> str:
        return html.unescape(text)

    def _remove_html_tags(self, text: str) -> str:
        pattern = re.compile(r"<.*?>")
        return re.sub(pattern, "", text)

    def clean(self, text: str) -> str:

        # clean pantip component
        text = self._replace_spoil_component(text)
        text = self._replace_edit_text(text)
        text = self._replace_pantip_spaces(text)
        text = self._unescape_html(text)
        text = self._remove_html_tags(text)

        # call parent func
        text = super().clean(text)
        return text