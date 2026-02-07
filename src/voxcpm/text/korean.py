# reference: https://github.com/ORI-Muchim/MB-iSTFT-VITS-Korean/blob/main/text/korean.py

import re
from .lib.ko.g2pk4.jamo import h2j, j2hcj
from .lib.ko.g2pk4 import ko_pron
from .lib.ko.g2pk4 import G2p


from .symbols2 import symbols



# This is a list of Korean classifiers preceded by pure Korean numerals.
_korean_classifiers = (
    "군데 권 개 그루 닢 대 두 마리 모 모금 뭇 발 발짝 방 번 벌 보루 살 수 술 시 쌈 움큼 정 짝 채 척 첩 축 켤레 톨 통"
)

# List of (hangul, hangul divided) pairs:
_hangul_divided = [
    (re.compile("%s" % x[0]), x[1])
    for x in [
        # ('ㄳ', 'ㄱㅅ'),   # g2pk2, A Syllable-ending Rule
        # ('ㄵ', 'ㄴㅈ'),
        # ('ㄶ', 'ㄴㅎ'),
        # ('ㄺ', 'ㄹㄱ'),
        # ('ㄻ', 'ㄹㅁ'),
        # ('ㄼ', 'ㄹㅂ'),
        # ('ㄽ', 'ㄹㅅ'),
        # ('ㄾ', 'ㄹㅌ'),
        # ('ㄿ', 'ㄹㅍ'),
        # ('ㅀ', 'ㄹㅎ'),
        # ('ㅄ', 'ㅂㅅ'),
        ("ㅘ", "ㅗㅏ"),
        ("ㅙ", "ㅗㅐ"),
        ("ㅚ", "ㅗㅣ"),
        ("ㅝ", "ㅜㅓ"),
        ("ㅞ", "ㅜㅔ"),
        ("ㅟ", "ㅜㅣ"),
        ("ㅢ", "ㅡㅣ"),
        ("ㅑ", "ㅣㅏ"),
        ("ㅒ", "ㅣㅐ"),
        ("ㅕ", "ㅣㅓ"),
        ("ㅖ", "ㅣㅔ"),
        ("ㅛ", "ㅣㅗ"),
        ("ㅠ", "ㅣㅜ"),
    ]
]

# List of (Latin alphabet, hangul) pairs:
_latin_to_hangul = [
    (re.compile("%s" % x[0], re.IGNORECASE), x[1])
    for x in [
        ("a", "에이"),
        ("b", "비"),
        ("c", "시"),
        ("d", "디"),
        ("e", "이"),
        ("f", "에프"),
        ("g", "지"),
        ("h", "에이치"),
        ("i", "아이"),
        ("j", "제이"),
        ("k", "케이"),
        ("l", "엘"),
        ("m", "엠"),
        ("n", "엔"),
        ("o", "오"),
        ("p", "피"),
        ("q", "큐"),
        ("r", "아르"),
        ("s", "에스"),
        ("t", "티"),
        ("u", "유"),
        ("v", "브이"),
        ("w", "더블유"),
        ("x", "엑스"),
        ("y", "와이"),
        ("z", "제트"),
    ]
]

# List of (ipa, lazy ipa) pairs:
_ipa_to_lazy_ipa = [
    (re.compile("%s" % x[0], re.IGNORECASE), x[1])
    for x in [
        ("t͡ɕ", "ʧ"),
        ("d͡ʑ", "ʥ"),
        ("ɲ", "n^"),
        ("ɕ", "ʃ"),
        ("ʷ", "w"),
        ("ɭ", "l`"),
        ("ʎ", "ɾ"),
        ("ɣ", "ŋ"),
        ("ɰ", "ɯ"),
        ("ʝ", "j"),
        ("ʌ", "ə"),
        ("ɡ", "g"),
        ("\u031a", "#"),
        ("\u0348", "="),
        ("\u031e", ""),
        ("\u0320", ""),
        ("\u0339", ""),
    ]
]

def latin_to_hangul(text):
    for regex, replacement in _latin_to_hangul:
        text = re.sub(regex, replacement, text)
    return text


def divide_hangul(text):
    text = j2hcj(h2j(text))
    for regex, replacement in _hangul_divided:
        text = re.sub(regex, replacement, text)
    return text


def korean_to_lazy_ipa(text):
    text = latin_to_hangul(text)
    #text = number_to_hangul(text)
    text = re.sub("[\uac00-\ud7af]+", lambda x: ko_pron.romanise(x.group(0), "ipa").split("] ~ [")[0], text)
    for regex, replacement in _ipa_to_lazy_ipa:
        text = re.sub(regex, replacement, text)
    return text


_g2p = G2p(tokenizer_model="./checkpoints/1.5-large-ko-rev-a/KoichiYasuoka__roberta-large-korean-morph-upos")


def korean_to_ipa(text):
    text = latin_to_hangul(text)
    #text = number_to_hangul(text)
    text = _g2p(text)
    #text = fix_g2pk2_error(text)
    text = korean_to_lazy_ipa(text)
    return text.replace("ʧ", "tʃ").replace("ʥ", "dʑ")


def post_replace_ph(ph):
    rep_map = {
        "：": ",",
        "；": ",",
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "\n": ".",
        "·": ",",
        "、": ",",
        "...": "…",
        " ": "空",
    }
    if ph in rep_map.keys():
        ph = rep_map[ph]
    if ph in symbols:
        return ph
    if ph not in symbols:
        ph = "停"
    return ph


def g2p(text):
    text = latin_to_hangul(text)
    text = _g2p(text)
    text = divide_hangul(text)
    #text = fix_g2pk2_error(text)
    text = re.sub(r"([\u3131-\u3163])$", r"\1.", text)
    text = "".join([post_replace_ph(i) for i in text])
    #text = [post_replace_ph(i) for i in text]
    return text


if __name__ == "__main__":
    text = "안녕하세요"
    print(g2p(text))
