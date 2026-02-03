from .lib.ja.voicevox_g2p import pyopenjtalk
from .lib.ja.voicevox_g2p.kana_converter import create_kana
from .lib.ja.voicevox_g2p.text_analyzer import full_context_labels_to_accent_phrases

def g2p(norm_text) -> str:
    full_context_labels = pyopenjtalk.extract_fullcontext(norm_text)
    accent_phrases = full_context_labels_to_accent_phrases(full_context_labels)
    kana = create_kana(accent_phrases)
    return kana
