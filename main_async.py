import asyncio
import time
import os

from corgo.input_handlers import get_user_query
from corgo.agents import OpenWebUIAgent
from corgo.tts_models import OpenAISTTModel
from corgo.audio_utils import play_audio


async def main():
    # Get user query
    t = time.time()
    query = await get_user_query()
    print(f"Received input: {query} in {time.time() - t:.2f} sec")

    # Get agent response
    t = time.time()
    agent = OpenWebUIAgent()
    response = agent.get_response(query)
    print(f"Response: {response}")
    print(f"Time taken = {time.time() - t:.2f} sec")

    # Convert response to audio
    t = time.time()
    output_path = f"agent_reponse.mp3"
    model_tts = OpenAISTTModel()
    model_tts.generate_audio(response, output_path)
    print(f"Time taken = {time.time() - t:.2f} sec")

    # Play agent response
    t = time.time()
    play_audio(output_path)
    print(f"Played audio in {time.time() - t:.2f} sec")

    # cleanup
    os.remove(output_path)


if __name__ == "__main__":
    while True:
        print("Waiting for input ...")
        asyncio.run(main())    