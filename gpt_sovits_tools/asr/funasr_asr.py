# -*- coding:utf-8 -*-

import argparse
import os
import traceback

# from funasr.utils import version_checker
# version_checker.check_for_update = lambda: None
from funasr import AutoModel
from tqdm import tqdm

funasr_models = {}  # 存储模型避免重复加载

from voxcpm.text.auto import G2p




def only_asr(input_file, language):
    try:
        model = create_model(language)
        text = model.generate(input=input_file)[0]["text"]
    except:
        text = ""
        print(traceback.format_exc())
    return text


def create_model(language="zh"):

    path_asr = "./checkpoints/SenseVoiceSmall"
    model_revision = None
    vad_model_revision = None
    path_vad = None
    path_punc = None
    punc_model_revision = None

    if language in funasr_models:
        return funasr_models[language]
    else:
        model = AutoModel(
            model=path_asr,
            model_revision=model_revision,
            vad_model=path_vad,
            vad_model_revision=vad_model_revision,
            punc_model=path_punc,
            punc_model_revision=punc_model_revision,
        )
        print(f"FunASR 模型加载完成: {language.upper()}")

        funasr_models[language] = model
        return model


def execute_asr(input_folder, output_folder, model_size, language):
    input_file_names = os.listdir(input_folder)
    input_file_names.sort()

    output = []
    output_file_name = os.path.basename(input_folder)

    model = create_model(language)
    g2p = G2p()

    for file_name in tqdm(input_file_names):
        try:
            print("\n" + file_name)
            file_path = os.path.join(input_folder, file_name)
            text = model.generate(input=file_path)[0]["text"]
            text = text.split('|>')[-1]
            text = g2p(text)
            output.append("{" + f'"audio": "{file_path}", "text": "{text}"' + "}")
        except:
            print(traceback.format_exc())

    output_folder = output_folder or "output/asr_opt"
    os.makedirs(output_folder, exist_ok=True)
    output_file_path = os.path.abspath(f"{output_folder}/{output_file_name}.jsonl")

    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output))
        print(f"ASR 任务完成->标注文件路径: {output_file_path}\n")
    return output_file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_folder", type=str, required=True, help="Path to the folder containing WAV files."
    )
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Output folder to store transcriptions.")
    parser.add_argument("-s", "--model_size", type=str, default="large", help="Model Size of FunASR is Large")
    parser.add_argument(
        "-l", "--language", type=str, default="zh", choices=["zh", "yue", "auto"], help="Language of the audio files."
    )
    parser.add_argument(
        "-p", "--precision", type=str, default="float16", choices=["float16", "float32"], help="fp16 or fp32"
    )  # 还没接入
    cmd = parser.parse_args()
    execute_asr(
        input_folder=cmd.input_folder,
        output_folder=cmd.output_folder,
        model_size=cmd.model_size,
        language=cmd.language,
    )
