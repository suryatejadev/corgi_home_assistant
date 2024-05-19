import os
from glob import glob
import time
import ffmpeg
from openai import OpenAI
from pygame import mixer
import wave
import sys
import json
from vosk import Model, KaldiRecognizer, SetLogLevel
SetLogLevel(-1)

PHONE_RECORDINGS_PATH = "./data/easy_voice_recorder"
client = OpenAI()
vosk_model = Model(model_name="vosk-model-small-en-in-0.4")

def convert_m4a_to_vosk(input_path, output_path):
    return ffmpeg.input(input_path).output(output_path, acodec="pcm_s16le", ar="16k", ac=1, loglevel="quiet", y=None).run()

def convert_m4a_to_mp3(input_path, output_path):
    return ffmpeg.input(input_path).output(output_path, loglevel="quiet").run()

def get_latest_recording():
    old_recording_files = glob(os.path.join(PHONE_RECORDINGS_PATH, "*.m4a"))    
    time.sleep(2)
    new_recording_files = glob(os.path.join(PHONE_RECORDINGS_PATH, "*.m4a"))
    diff_files = [k for k in new_recording_files if k not in old_recording_files]
    if len(diff_files) == 0:
        return
    latest_recording = sorted(diff_files, reverse=True)[0]
    return latest_recording

def speech_to_text(audio_file_path):
    with wave.open(audio_file_path, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            print("Audio file must be WAV format mono PCM.")
            sys.exit(1)

        # You can also init model by name or with a folder path
        # model = Model(model_name="vosk-model-en-us-0.21")
        # model = Model("models/en")

        rec = KaldiRecognizer(vosk_model, wf.getframerate())
        rec.SetWords(True)
        rec.SetPartialWords(True)

        results = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                results.append(rec.Result())
            # else:
            #     results.append(rec.PartialResult())

        results.append(rec.FinalResult())
    text = json.loads(results[0])["text"]
    return text

def get_chatgpt_response(input_path, output_path):
    ''' load audio file
    '''
    t = time.time()
    with open(input_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        ).text
    print(f"Transcribed in {time.time() - t:.2f} sec")
    print(f"Input question: {transcription}")

    # t = time.time()
    # transcription = speech_to_text(input_path)
    # print(f"Transcribed in {time.time() - t:.2f} sec")
    # print(f"Input question: {transcription}")

    '''Generate answer
    '''
    t = time.time()
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are an assistant. Always generate short, succint responses, in 1 to 2 lines."},
        {"role": "user", "content": transcription}
    ]).choices[0].message.content
    print(f"Generated response in {time.time() - t:.2f} sec")
    print(f"ChatGPT Response: {completion}")

    '''Convert answer to speech
    '''    
    t = time.time()
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=completion
    )
    response.stream_to_file(output_path)
    print(f"Generated audio file in {time.time() - t:.2f} sec")

def play_audio(audio_path):
    mixer.init()
    mixer.music.load(audio_path)
    mixer.music.play()
    while mixer.music.get_busy():
        time.sleep(1)

def run():
    latest_recording = get_latest_recording()    
    if latest_recording is None:
        return    
    
    print("Got recording ...")
    recording_name = os.path.basename(latest_recording).split(".")[0]
    latest_recording_mp3 = f"data/easy_voice_recorder_mp3/{recording_name}.mp3"
    t = time.time()
    convert_m4a_to_mp3(
        input_path=latest_recording,
        output_path=latest_recording_mp3
        )
    print(f"Converted to mp3 in {time.time() - t:.2f} sec")

    # latest_recording_wav = f"data/easy_voice_recorder_wav/{recording_name}.wav"
    # t = time.time()
    # convert_m4a_to_vosk(
    #     input_path=latest_recording,
    #     output_path=latest_recording_wav
    #     )
    # print(f"Converted to wav in {time.time() - t:.2f} sec")
        
    output_path = f"data/chatgpt_responses/{recording_name}.mp3"
    get_chatgpt_response(
        input_path=latest_recording_mp3,
        output_path=output_path
        )
    t = time.time()
    play_audio(output_path)
    print(f"Played audio in {time.time() - t:.2f} sec")

if __name__ == "__main__":
    # run()
    while 1:
        run()