import requests


class Agent:
    def get_response(self, query):
        raise NotImplementedError

class OpenWebUIAgent(Agent):
    def __init__(self):
        token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6Ijc3ZDFiYjAwLTFiMzQtNDc1MC1hYTA0LTdiM2JhMGI1YTgxMSJ9.RY3qTPFkUH7BmU6Nld1rYNCn9L8tLFP-k7Xx0fhEeH4"
        self.server_url = "http://10.0.0.84:8080/api/chat/completions"
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
        self.model_name = "llama3.2:3b"

    def get_response(self, query):
        # Prepare the request payload
        payload = {"model": self.model_name, "messages": [{"role": "user", "content": query}]}

        # Send the POST request
        try:
            response = requests.post(self.server_url, json=payload, headers=self.headers)
            response.raise_for_status()  # Raise an error for bad responses (4xx and 5xx)
            response = response.json()["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            import traceback; traceback.print_exc()
            response = None
        
        return response