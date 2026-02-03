# MIT License

# Copyright (c) 2024 RVC-Boss

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os
from pathlib import Path
import pickle

EN_DICT_DIR = Path(__file__).parent / "en_dict"
CMU_DICT_PATH = str(EN_DICT_DIR / "cmudict.rep")
CMU_DICT_FAST_PATH = str(EN_DICT_DIR / "cmudict-fast.rep")
CMU_DICT_HOT_PATH = str(EN_DICT_DIR /"engdict-hot.rep")
CACHE_PATH = str(EN_DICT_DIR / "engdict_cache.pickle")
NAMECACHE_PATH = str(EN_DICT_DIR / "namedict_cache.pickle")

def read_dict_new():
    g2p_dict = {}
    with open(CMU_DICT_PATH) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= 57:
                line = line.strip()
                word_split = line.split("  ")
                word = word_split[0].lower()
                g2p_dict[word] = [word_split[1].split(" ")]

            line_index = line_index + 1
            line = f.readline()

    with open(CMU_DICT_FAST_PATH) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= 0:
                line = line.strip()
                word_split = line.split(" ")
                word = word_split[0].lower()
                if word not in g2p_dict:
                    g2p_dict[word] = [word_split[1:]]

            line_index = line_index + 1
            line = f.readline()

    return g2p_dict


def hot_reload_hot(g2p_dict):
    with open(CMU_DICT_HOT_PATH) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= 0:
                line = line.strip()
                word_split = line.split(" ")
                word = word_split[0].lower()
                # 自定义发音词直接覆盖字典
                g2p_dict[word] = [word_split[1:]]

            line_index = line_index + 1
            line = f.readline()

    return g2p_dict

def cache_dict(g2p_dict, file_path):
    with open(file_path, "wb") as pickle_file:
        pickle.dump(g2p_dict, pickle_file)

def get_dict():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as pickle_file:
            g2p_dict = pickle.load(pickle_file)
    else:
        g2p_dict = read_dict_new()
        cache_dict(g2p_dict, CACHE_PATH)

    g2p_dict = hot_reload_hot(g2p_dict)

    return g2p_dict

# def read_dict():
#     g2p_dict = {}
#     start_line = 49
#     with open(CMU_DICT_PATH) as f:
#         line = f.readline()
#         line_index = 1
#         while line:
#             if line_index >= start_line:
#                 line = line.strip()
#                 word_split = line.split("  ")
#                 word = word_split[0].lower()

#                 syllable_split = word_split[1].split(" - ")
#                 g2p_dict[word] = []
#                 for syllable in syllable_split:
#                     phone_split = syllable.split(" ")
#                     g2p_dict[word].append(phone_split)

#             line_index = line_index + 1
#             line = f.readline()

#     return g2p_dict