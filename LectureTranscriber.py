import sounddevice
import whisper
from scipy.signal import resample_poly
from math import gcd
from time import perf_counter
from datetime import datetime, timedelta
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

def audio_callback(indata, frames, time_info, status):
    if status.input_overflow:
        input_overflow.set()
        
    try:
        audio_queue.put_nowait(indata.copy())
    except Full:
        queue_full.set()
        raise sounddevice.CallbackAbort

recording_started = datetime.now()
date_text = recording_started.strftime("%Y/%m/%d")

markdown_text = (
    f"# {lecture_name}\n\n"
    f"{date_text}\n\n"
)

blocks = []
buffered_frames = 0
processed_frames = 0

# 複数のブロックをまとめて文字起こしする関数
def process_blocks(blocks, processed_frames):
    audio = numpy.concatenate(blocks, axis=0)
    
    # 録音開始時刻 + 処理済みの音声の長さ
    chunk_started = recording_started + timedelta(
        seconds = processed_frames / sample_rate
    )
    
    text, elapsed = transcribe_audio(audio, sample_rate, model)
    time_text = chunk_started.strftime("%H:%M:%S")
    
    print(f"\n{time_text}")
    print(text)
    print(f"処理時間: {elapsed:.2f}秒")
    
    markdown_part = (
        f"## {time_text}\n\n"
        f"{text}\n\n"
    )
    
    return markdown_part, len(audio)


# main
print("モデル読み込み中")
model = whisper.load_model("base", device="cpu")

sample_rate = int(sounddevice.query_devices(kind="input")["default_samplerate"]) #マイクの既定のサンプルレートにする


#役0.5秒分ずつ音声を受け取る
block_frames = int(sample_rate * 0.5)

# 5秒分たまったら文字起こしする
chunk_frames = int(sample_rate * 5)

# 最大120ブロック(役50秒分)を待機させる
audio_queue = Queue(maxsize=120)
queue_full = Event()
input_overflow = Event()



print("録音開始。Ctrl + Cで停止・保存します。")
try:
    with sounddevice.InputStream(
        samplerate = sample_rate,
        channels=1,
        dtype="float32",
        blocksize=block_frames,
        callback=audio_callback,
    ):
        while not queue_full.is_set():
            try:
                block = audio_queue.get(timeout=0.2)
            except Empty:
                continue
            
            blocks.append(block)
            buffered_frames += len(block)
            
            if buffered_frames >= chunk_frames:
                part, frames = process_blocks(
                    blocks, processed_frames
                )
                
                markdown_text += part
                processed_frames += frames
                
                blocks = []
                buffered_frames = 0

except KeyboardInterrupt:
    print(f"\n録音を停止します")
    
if queue_full.is_set():
    print("未処理音声が上限に達したため、録音を停止しました")
    
if input_overflow.is_set():
    print(f"警告: マイク入力で音声の欠落が発生しました")
    
# 録音停止後、キュ０に残った音声も処理する
print("残りの音声を処理中")

while True:
    try:
        block = audio_queue.get_nowait()
    except Empty:
        break
    
    blocks.append(block)
    buffered_frames += len(block)

    if buffered_frames >= chunk_frames:
        part, frames = process_blocks(blocks, processed_frames)
        markdown_text += part
        processed_frames += frames

        blocks = []
        buffered_frames = 0

# 最後の5秒未満の音声も処理する
if blocks:
    part, frames = process_blocks(blocks, processed_frames)
    markdown_text += part


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


