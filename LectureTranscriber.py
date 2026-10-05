import sounddevice
import whisper
from scipy.signal import resample_poly
from math import gcd
from time import perf_counter
from datetime import datetime, timedalta
import sys
import numpy
from queue import Queue, Empty, Full
from threading import Event

# コマンドライン引数処理
if len(sys.argv) != 2:
    print(f'使い方：python3 LectureTranscriber.py "授業名"')
    sys.exit(1)
filename_arg = sys.argv[1]
if not filename_arg.endswith(".md"):
    print(f"ファイル名の末尾を .md にしてください。")
    sys.exit(1)
lecture_name = filename_arg[:-3]
if not lecture_name.strip(): # 戦闘と末尾の空白を取り除く
    print("授業名を指定してください")
    sys.exit(1)


def transcribe_audio(audio, sample_rate, model):
    # (サンプル数, 1)の配列を1次元配列にする
    audio_mono = audio[:, 0]

    # マイクのサンプルレートから16000Hzへ変換する
    target_rate = 16000
    divisor = gcd(sample_rate, target_rate)

    audio_16k = resample_poly(
        audio_mono,
        up=target_rate // divisor,
        down=sample_rate // divisor,
    ).astype("float32")

    started = perf_counter()

    result = model.transcribe(
        audio_16k,
        language="ja",
        task="transcribe",
        fp16=False,
    )

    elapsed = perf_counter() - started
    return result["text"].strip(), elapsed

print("モデル読み込み中")
model = whisper.load_model("base", device="cpu")

sample_rate = int(sounddevice.query_devices(kind="input")["default_samplerate"]) #マイクの既定のサンプルレートにする
# duration = 5 #録音時間
# print("録音開始")
# recording_started = datetime.now()
# audio = sounddevice.rec(
#     frames=int(sample_rate * duration),
#     samplerate=sample_rate,
#     channels=1, #モノラル
#     dtype="float32",
# )

# sounddevice.wait()

# print("再生")
# sounddevice.play(audio, samplerate=sample_rate)
# sounddevice.wait()

# print("文字起こし中")
# text, elapsed = transcribe_audio(audio, sample_rate, model)


# print(f"\n文字起こし結果:")
# print(recording_started.strftime("%H:%M:%S"))
# print(text)
# print(f"\n処理時間: {elapsed:2f}秒")


# date_text = recording_started.strftime("%Y/%m/%d")
# time_text = recording_started.strftime("%H:%M:%S")


# markdown_text = (
#     f"# {lecture_name}\n"
#     f"{date_text}\n\n"
#     f"## {time_text}\n"
#     f"{text}\n"
# )

# 同名ファイルが有れば更新をつける


base_name = f"{lecture_name}"
number = 0

while True:
    if number == 0:
        filename = f"{base_name}.md"
    else:
        filename = f"{base_name}_{number}.md"
    
    try:
        with open(filename, mode="x", encoding="utf-8") as file:
            file.write(markdown_text)
        break

    except FileExistsError:
        number += 1

print(f"保存しました: {filename}")


