import pygame
import simpleaudio as sa
import numpy as np
import time
import csv

def generate_beep(frequency=440, duration=0.1, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(frequency * 2 * np.pi * t)
    audio = tone * (2**15 - 1) / np.max(np.abs(tone))
    audio = audio.astype(np.int16)
    return audio

def read_timestamps_from_csv(file_path):
    timestamps = []
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            timestamps.append(float(row[0]))
    return timestamps

def play_music_with_beeps(music_file, timestamps):
    pygame.mixer.init()
    pygame.mixer.music.load(music_file)
    pygame.mixer.music.play()

    beep_audio = generate_beep()
    sample_rate = 44100

    start_time = time.time()
    idx = 0
    total = len(timestamps)

    while pygame.mixer.music.get_busy():
        current_time = time.time() - start_time

        # 播放所有时间戳中时间<=当前时间的滴声
        while idx < total and timestamps[idx] <= current_time:
            sa.play_buffer(beep_audio, 1, 2, sample_rate)
            print(f"滴声 @ {timestamps[idx]:.3f} 秒")
            idx += 1

        time.sleep(0.005)  # 5ms轮询，降低CPU占用

    print("音乐播放结束")

if __name__ == "__main__":
    music_path = "/home/hy/wust_vision/fun/Gala - 你_H.ogg"  # 音乐路径
    timestamps_csv = "/home/hy/wust_vision/user_timestamps.csv"  # 时间戳路径

    timestamps = read_timestamps_from_csv(timestamps_csv)
    print(f"读取到 {len(timestamps)} 个时间戳")

    play_music_with_beeps(music_path, timestamps)
