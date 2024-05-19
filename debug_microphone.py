import speech_recognition as sr
import sounddevice
import time


def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("speak")
        audio = r.listen(source, phrase_time_limit=5)
        print("heard")
        said = ""
        try:
            t = time.time()
            # said = r.recognize_google(audio)
            # said = r.recognize_sphinx(audio)
            said = r.recognize_whisper(audio, language="english")
            print(f"you said = {said}, transcribed in {time.time()-t:.2f} sec")
        except Exception as e:
            print("Exception: " + str(e))
    return said

said = get_audio()
