import google.generativeai as genai
from django.conf import settings

class UnifiedChatClient:
    def __init__(self, api_key=None, model=None, generation_config=None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = model or settings.CHAT_MODEL
        self.generation_config = generation_config or settings.GENERATION_CONFIG

        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(
            model_name=self.model,
            generation_config=self.generation_config
        )

    def chat(self, messages):
        # Baue Prompt aus den Nachrichten
        prompt = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in messages
        )

        # Anfrage an Gemini senden
        response = self.client.generate_content(prompt)

        return response.text.strip()
