import os
from glob import glob
import time
import ffmpeg
from openai import OpenAI
from pygame import mixer
import speech_recognition as sr
import sounddevice


client = OpenAI()
PHONE_RECORDINGS_PATH = "./data/easy_voice_recorder"


def get_chatgpt_response(query, output_path):
    # Generate answer
    t = time.time()
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are an assistant. Always generate short, succint responses, in 1 to 2 lines."},
        {"role": "user", "content": query}
    ]).choices[0].message.content
    print(f"Generated response in {time.time() - t:.2f} sec")
    print(f"ChatGPT Response: {completion}")

    # Convert answer to speech
    t = time.time()
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=completion
    )
    response.stream_to_file(output_path)
    print(f"Generated audio file in {time.time() - t:.2f} sec")

def play_audio(audio_path):
    mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
    mixer.music.load(audio_path)
    mixer.music.play()
    while mixer.music.get_busy():
        time.sleep(1)

def get_latest_recording():
    available_files = glob(os.path.join(PHONE_RECORDINGS_PATH, "*.m4a"))
    if len(available_files) == 0:
        return
    latest_recording = sorted(available_files, reverse=True)[0]
    return latest_recording

def convert_m4a_to_mp3(input_path, output_path):
    return ffmpeg.input(input_path).output(output_path, loglevel="quiet").run()

def get_user_audio_query_from_phone():
    # check if any new files are added to the phone-synced folder
    latest_recording = get_latest_recording()

    if latest_recording is None:
        return

    # acknowledge
    print("Got recording ...")
    play_audio("data/sound_effects/corgi_bark.mp3")

    # convert file to mp3
    latest_recording_mp3 = "tmp_query.mp3"
    t = time.time()
    convert_m4a_to_mp3(
        input_path=latest_recording,
        output_path=latest_recording_mp3
        )
    print(f"Converted to mp3 in {time.time() - t:.2f} sec")

    # transcribe to get query
    with open(latest_recording_mp3, "rb") as audio_file:
            query = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            ).text

    # cleanup tmp data and remove all files in phone-synced folder
    os.remove(latest_recording_mp3)
    for path in glob(os.path.join(PHONE_RECORDINGS_PATH, "*.m4a")):
        os.remove(path)

    return query

def get_user_audio_query_from_microphone(hotword=False):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source, phrase_time_limit=2 if hotword else 5)
        said = ""
        try:
            t = time.time()
            said = r.recognize_google(audio).lower()
            if not hotword:
                return said
            if "corgi" in said or "corgee" in said or "corgy" in said or "corgo" in said or "cargo" in said:
                play_audio("data/sound_effects/corgi_bark.mp3")
                return True
        except Exception as e:
            pass
    return

def get_user_audio_query():
    query_from_phone = get_user_audio_query_from_phone()
    if query_from_phone is not None:
        return query_from_phone
    if get_user_audio_query_from_microphone(hotword=True):
        return get_user_audio_query_from_microphone(hotword=False)

def run():
    # Get user query
    user_audio_query = get_user_audio_query()
    if user_audio_query is None:
        return
    print(f"Input question: {user_audio_query}")

    # Get agent response
    recording_name = "default"
    output_path = f"agent_reponse.mp3"
    get_chatgpt_response(
        query=user_audio_query,
        output_path=output_path
        )

    # Play agent response
    t = time.time()
    play_audio(output_path)
    print(f"Played audio in {time.time() - t:.2f} sec")

    # delete agent response
    os.remove(output_path)

if __name__ == "__main__":
    while 1:
        run()