from pygame import mixer
import time


def play_audio(audio_path):
    mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
    mixer.music.load(audio_path)
    mixer.music.play()
    while mixer.music.get_busy():
        time.sleep(1)        