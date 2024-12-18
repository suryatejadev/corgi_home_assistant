import asyncio
import inspect
import sys
import speech_recognition as sr
import sounddevice  # needed for sr to work
import time

from .audio_utils import play_audio


class AudioInputHandler:
    async def get_user_query(self):
        raise NotImplementedError("This method should be implemented by subclasses")

    async def get_user_query_loop(self):
        while True:
            result = await self.get_user_query()
            if result is not None:
                return result


class MicrophoneAudioInputHandler(AudioInputHandler):
    def __init__(self):
        self.hotwords = ["corgi", "corgee", "corgy", "corgo", "cargo"]
        self.bark_path = "/home/sdevarakonda/projects/repos/corgi_home_assistant/data/sound_effects/corgi_bark.mp3"

    async def keyword_detection(self):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            try:
                audio = await asyncio.to_thread(r.listen, source, phrase_time_limit=2)
                said = await asyncio.to_thread(r.recognize_google, audio)
                said = said.lower()

                if any(hotword in said for hotword in self.hotwords):                
                    await asyncio.to_thread(play_audio, self.bark_path)
                    return True
            except sr.UnknownValueError:
                # Ignore if speech recognition couldn't understand the audio
                pass
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"Error in keyword detection: {e}")           
        return False

    async def get_query_from_mic(self):
        print('listening for query')
        r = sr.Recognizer()
        with sr.Microphone() as source:
            try:
                audio = r.listen(source, phrase_time_limit=5)
                said = await asyncio.to_thread(r.recognize_google, audio)
                said = said.lower()                
                return said
            except sr.UnknownValueError:
                # Ignore if speech recognition couldn't understand the audio
                pass
            except Exception as e:
                print(f"Error in getting query: {e}")
        return None

    async def get_user_query(self):
        if await self.keyword_detection():
            return await self.get_query_from_mic()


async def get_user_query():
    # Dynamically discover all subclasses of AudioInputHandler
    subclasses = [cls() for cls in AudioInputHandler.__subclasses__()]

    # Run all input functions concurrently
    tasks = [asyncio.create_task(handler.get_user_query_loop()) for handler in subclasses]

    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

    # Collect results from completed tasks
    results = [task.result() for task in done]

    # Cancel any pending tasks
    for task in pending:
        task.cancel()

    # Filter out None results
    non_none_results = [result for result in results if result is not None]

    if len(non_none_results) > 1:
        print("Multiple inputs received. Restarting listening...")
    elif len(non_none_results) == 1:
        return non_none_results[0]