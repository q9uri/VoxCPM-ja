# 例: Python/Hugging Face transformersの場合

# トークナイザをロード
from pathlib import Path
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("efwkjn/whisper-ja-anime-v0.3")

# 語彙を取得
vocab = tokenizer.get_vocab()

out = []
#https://sites.google.com/view/kilin/lecture/UTF-8table
"\u1D00-\u1D7F" #128文字	音声記号拡張* 添字も含む 	ᴀ ᴁ ᴂ ᴃ ᴄ ᴅ ᴆ ᴇ ᴈ ᴉ ᴊ ᴋ ᴌ ᴍ ᴎ ᴏ
"\u1D80-\u1DBF" #64文字	音声記号拡張補助* 	ᶀ ᶁ ᶂ ᶃ ᶄ ᶅ ᶆ ᶇ ᶈ ᶉ ᶊ ᶋ ᶌ ᶍ ᶎ ᶏ

GREEK_ALPHABET = "\u0370-\u03FF" #144文字	ギリシア文字及びコプト文字* 	α β γ δ ε ζ η θ ι κ λ μ ν ξ ο π
GREEK_ALPHABET_ADDITIONAL = "\u1F00-\u1FFF" # 256文字	ギリシア文字拡張 	ἀ ἁ ἂ ἃ ἄ ἅ ἆ ἇ Ἀ Ἁ Ἂ Ἃ Ἄ Ἅ

JAPANESE_HIRAGANA = "\u3040-\u309F" #96文字	平仮名 	ぁ あ ぃ い ぅ う ぇ え ぉ お か が き ぎ く ぐ
JAPANESE_KATAKANA = "\u30A0-\u30FF" #96文字	片仮名 	ァ ア ィ イ ゥ ウ ェ エ ォ オ カ ガ キ ギ ク グ

#CJK漢字
CJK_KANJI01 = "\u4E00-\u5FFF"  #4608文字 CJK統合漢字 	一 丁 丂 七 丄 丅 丆 万 丈 三 上 下 丌 不 与 丏
CJK_KANJI02 = "\u6000-\u7FFF" #8192文字 CJK統合漢字 	怀 态 怂 怃 怄 怅 怆 怇 怈 怉 怊 怋 怌 怍 怎 怏
CJK_KANJI03 = "\u8000-\u9FFF" #8192文字 CJK統合漢字 	耀 老 耂 考 耄 者 耆 耇 耈 耉 耊 耋 而 耍 耎 耏

CJK_KANJI_ADDITIONAL_A = "\u3400-\u3FFF" #3072文字	CJK統合漢字拡張A 	㐀 㐁 㐂 㐃 㐄 㐅 㐆 㐇 㐈 㐉 㐊 㐋 㐌 㐍 㐎 㐏
CJK_KANJI_ADDITIONAL_B = "\u020000-\u02A6DF" #42720	CJK統合漢字拡張B 	𠀀 𠀁 𠀂 𠀃 𠀄 𠀅 𠀆 𠀇 𠀈 𠀉 𠀊 𠀋 𠀌 𠀍 𠀎 𠀏
CJK_KANJI_ADDITIONAL_C = "\u02A700-\u02B738" #4152文字	CJK統合漢字拡張C 	𪜀 𪜁 𪜂 𪜃 𪜄 𪜅 𪜆 𪜇 𪜈 𪜉 𪜊 𪜋 𪜌 𪜍 𪜎 𪜏
CJK_KANJI_ADDITIONAL_D = "\u02B740-\u02B81D" #222文字	CJK統合漢字拡張D 	𫝀 𫝁 𫝂 𫝃 𫝄 𫝅 𫝆 𫝇 𫝈 𫝉 𫝊 𫝋 𫝌 𫝍 𫝎 𫝏
CJK_KANJI_ADDITIONAL_E = "\u02B820-\u02CEA1" #5762文字	CJK統合漢字拡張E
CJK_KANJI_ADDITIONAL_F = "\u02CEB0-\u02EBE0" #7473文字	CJK統合漢字拡張F
CJK_KANJI_ADDITIONAL_G = "\u030000-\u03134A" #2891文字	CJK統合漢字拡張G

CJK_KANJI_GOKAN = "\uF900-\uFAFF" #512文字	CJK互換漢字 	豈 更 車 賈 滑 串 句 龜 龜 契 金 喇 奈 懶 癩 羅
CJK_KANJI_GOKAN_SUPPORT = "\u02F800-\u02FA1F" #544文字	CJK互換漢字補助 	丽 丸 乁 𠄢 你 侮 侻 倂 偺 備 僧 像 㒞 𠘺 免 兔 =

SYMBOLS_EMOJI = "\u01F300-\u01F5FF" #その他の記号及び絵文字* 	🌀 🌁 🌂 🌃 🌄 🌅 🌆 🌇 🌈 🌉 🌊 🌋 🌌 🌍 🌎 🌏
SYMBOLS_MAJANG = "\u01F000-\u01F02F" #48文字	麻雀牌 	🀀 🀁 🀂 🀃 🀄 🀅 🀆 🀇 🀈 🀉 🀊 🀋 🀌 🀍 🀎 🀏
SYMBOLS_TRUMP_CARD = "\u01F0A0-\u01F0FF" #96文字	トランプ 	🂠 🂡 🂢 🂣 🂤 🂥 🂦 🂧 🂨 🂩 🂪 🂫 🂬 🂭 🂮
SYMBOLS_MONEY = "\u20A0-\u20CF" #48文字	通貨記号 	#₠ ₡ ₢ ₣ ₤ ₥ ₦ ₧ ₨ ₩ ₪ ₫ € ₭ ₮ ₯

RENKINJUTSU = "\u01F700-\u01F77F" #128文字 錬金術記号    🜀 🜁 🜂 🜃 🜄 🜅 🜆 🜇 🜈 🜉 🜊 🜋 🜌 🜍 🜎 🜏

SYMBOLS_NUM = "\u2200-\u22FF" #256文字	数学演算子* 	∀ ∁ ∂ ∃ ∄ ∅ ∆ ∇ ∈ ∉ ∊ ∋ ∌ ∍ ∎ ∏
SYMBOLS_NUM_A = "\u27C0-\u27EF" #48文字	その他の数学記号A* 	⟀ ⟁ ⟂ ⟃ ⟄ ⟅ ⟆ ⟇ ⟈ ⟉ ⟊ ⟋ ⟌ ⟍ ⟎ ⟏
SYMBOLS_NUM_B = "\u2980-\u29FF" #128文字 その他の数学記号B *    ⦀ ⦁ ⦂ ⦃ ⦄ ⦅ ⦆ ⦇ ⦈ ⦉ ⦊ ⦋ ⦌ ⦍ ⦎ ⦏
SYMBOLS_NUM_ALPHA = "\u01D400-\u01D7FF" #1024文字	数学用英数字記号* 	𝓐 𝓑 𝓒 𝓓 𝓔 𝓕 𝓖 𝓗 𝓘 𝓙 𝓚 𝓛 𝓜 𝓝 𝓞 𝓟

import re
KANJI_PATTERN = re.compile(rf"[{CJK_KANJI01}{CJK_KANJI02}{CJK_KANJI03}]+")
for _, token_id in vocab.items():
    # 1つのIDのみを含むリストをデコード
    decoded_string = tokenizer.decode([token_id])
    if KANJI_PATTERN.match(decoded_string):
        print(f"ID: {token_id}, 文字列: '{decoded_string}'")
        try:
            text = f"{decoded_string}\t{token_id}"
            out.append(text)
        except:
            print("not convertable")

Path("./tools/asr/whisper-ja-anime-v0.3.tsv").write_text("\n".join(out) ,encoding="utf-8")