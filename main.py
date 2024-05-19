import os
from glob import glob
import time
import ffmpeg
from openai import OpenAI
from pygame import mixer
import speech_recognition as sr
import sounddevice


client = OpenAI()

def get_chatgpt_response(input_path, output_path, transcribe=True):
    ''' load audio file
    '''
    t = time.time()
    if transcribe:
        with open(input_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            ).text
    else:
        transcription = input_path
    print(f"Transcribed in {time.time() - t:.2f} sec")
    print(f"Input question: {transcription}")

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

def convert_m4a_to_mp3(input_path, output_path):
    return ffmpeg.input(input_path).output(output_path, loglevel="quiet").run()

def get_latest_recording():
    PHONE_RECORDINGS_PATH = "./data/easy_voice_recorder"
    old_recording_files = glob(os.path.join(PHONE_RECORDINGS_PATH, "*.m4a"))    
    time.sleep(2)
    new_recording_files = glob(os.path.join(PHONE_RECORDINGS_PATH, "*.m4a"))
    diff_files = [k for k in new_recording_files if k not in old_recording_files]
    if len(diff_files) == 0:
        return
    latest_recording = sorted(diff_files, reverse=True)[0]
    return latest_recording

def get_user_audio_query_from_phone():
    latest_recording = get_latest_recording()    
    if latest_recording is None:
        return None, None   
    
    print("Got recording ...")
    play_audio("data/sound_effects/corgi_bark.mp3")
    recording_name = os.path.basename(latest_recording).split(".")[0]
    latest_recording_mp3 = f"data/easy_voice_recorder_mp3/{recording_name}.mp3"
    t = time.time()
    convert_m4a_to_mp3(
        input_path=latest_recording,
        output_path=latest_recording_mp3
        )
    print(f"Converted to mp3 in {time.time() - t:.2f} sec")
    return latest_recording_mp3, recording_name

def speech_recognition(hotword=False):    
    r = sr.Recognizer()
    while 1:
        with sr.Microphone() as source:
            # print("speak")                        
            audio = r.listen(source, phrase_time_limit=2 if hotword else 5)
            # print("heard")
            said = ""
            try:
                t = time.time()
                said = r.recognize_google(audio).lower()
                # print(f"you said = {said}, transcribed in {time.time()-t:.2f} sec")
                if not hotword:
                    return said
                if "corgi" in said or "corgee" in said or "corgy" in said or "corgo" in said or "cargo" in said:
                    play_audio("data/sound_effects/corgi_bark.mp3")
                    return
            except Exception as e:
                pass
                # print("Exception: " + str(e))
    return

def get_user_audio_query_from_microphone():
    print('Hotword detection...')
    speech_recognition(hotword=True)
    print('Query detection...')
    return speech_recognition(hotword=False)    

def get_user_audio_query(source="phone"):
    if source == "phone":
        return get_user_audio_query_from_phone()
    if source == "microphone":
        return get_user_audio_query_from_microphone(), None
    return

def run():
    # Get user query
    user_audio_query, recording_name = get_user_audio_query(source="microphone")
    if user_audio_query is None:
        return

    # Get agent response
    recording_name = recording_name or "default"
    output_path = f"data/chatgpt_responses/{recording_name}.mp3"
    get_chatgpt_response(
        input_path=user_audio_query,
        output_path=output_path,
        transcribe=False    
        )

    # Play agent response
    t = time.time()
    play_audio(output_path)
    print(f"Played audio in {time.time() - t:.2f} sec")

if __name__ == "__main__":
    while 1:
        run()