from openai import OpenAI



class STTModel:
    def generate_audio(self, output_path):
        raise NotImplementedError


class OpenAISTTModel:
    def __init__(self):
        self.client = OpenAI()
    
    def generate_audio(self, text, output_path):
        response = self.client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        response.stream_to_file(output_path)