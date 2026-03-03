
from .table import JAPANESE_PATTERN, KOREAN_PATTERN
import re2 as re

#KOREAN_PATTERN = f"[{HANGUL}[{HANGUL_JAMO}{HANGUL_JAMO_ADDITONAL_B}{HANGUL_JAMO_ADDITONAL_A}]"
#JAPANESE_PATTERN = f"[{JAPANESE_HIRAGANA}{JAPANESE_KATAKANA}]"

class G2p:
    def __init__(self):
        self.JAPANESE_PATTERN = re.compile(JAPANESE_PATTERN)
        self.KOREAN_PATTERN = re.compile(KOREAN_PATTERN)
        self.lang = "ja"
        self.g2p_ja = None
        self.g2p_ko = None
        self.g2p_zh = None

    def _lang_select(self, text):
        if self.JAPANESE_PATTERN.search(text):
            self.lang = "ja"

        elif self.KOREAN_PATTERN.search(text):
            self.lang = "ko"

        else:
            self.lang = "ja"

    def _load_g2p(self, lang):
        if lang == "ja" and self.g2p_ja == None:
            from .japanese import g2p as g2p_ja
            self.g2p_ja = g2p_ja

        elif lang == "ko" and self.g2p_ko == None:
            from .korean import g2p as g2p_ko
            self.g2p_ko = g2p_ko

        elif lang == "zh" and self.g2p_zh == None:
            from .chinese2 import g2p as g2p_zh
            self.g2p_zh = g2p_zh

    def _g2p(self, text, lang):
        if lang == "ja" and self.g2p_ja != None:
            return self.g2p_ja(text)

        elif lang == "ko" and self.g2p_ko != None:
            return self.g2p_ko(text)

        elif lang == "zh" and self.g2p_zh != None:
            return self.g2p_zh(text)

        else:
            return text

    def __call__(self, text):
        self._lang_select(text)
        self._load_g2p(self.lang)
        out = self._g2p(text, self.lang)

        return out
