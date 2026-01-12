from __future__ import annotations

import json
import requests


class AIKnowledge:
    def __init__(self, user_knowledge, assistant_knowledge):
        self.user_knowledge = user_knowledge
        self.assistant_knowledge = assistant_knowledge


class AIInterfaceException(Exception):
    pass


class AIInterface:
    MODEL_GPT_35 = "gpt-3.5-turbo"
    MODEL_GPT_41 = "gpt-4.1"

    NO_URL = "No URL specified"
    NO_KEY = "No key specified"
    NO_AI_MODEL = "No ai model specified (e.g. 'gpt-3.5-turbo')"
    CONNECTION_PB = "Connection to server failed"
    DISCONNECTION_PB = "DisConnection to server failed"

    def __init__(self):
        self.url_text = None
        self.key = None
        self.ai_model = self.MODEL_GPT_41
        self.knowledge = []

    @staticmethod
    def is_openai(model):
        return model.lower().startswith("gpt")

    @staticmethod
    def is_mistralai(model):
        return "tral" in model.lower()

    def set_url(self, url):
        self.url_text = url

    def set_key(self, key):
        self.key = key

    def has_key(self):
        return self.key is not None

    def set_ai_model(self, ai_model):
        self.ai_model = ai_model

    def clear_knowledge(self):
        self.knowledge.clear()

    def remove_previous_knowledge(self):
        if self.knowledge:
            self.knowledge.pop()

    def add_knowledge(self, user_knowledge, assistant_knowledge):
        self.knowledge.append(AIKnowledge(user_knowledge, assistant_knowledge))

    def get_knowledge(self):
        return self.knowledge

    def chat(self, text, use_knowledge_as_input=False, use_output_knowledge=False):
        _ = use_knowledge_as_input
        if not self.url_text:
            raise AIInterfaceException(self.NO_URL)
        if not self.key:
            raise AIInterfaceException(self.NO_KEY)
        if not self.ai_model:
            raise AIInterfaceException(self.NO_AI_MODEL)

        headers = {
            "Content-Type": "application/json; charset=UTF-8",
            "Authorization": f"Bearer {self.key}",
        }

        main_object = {"model": self.ai_model if self.ai_model else self.MODEL_GPT_35, "messages": []}

        main_object["messages"].append({"role": "system", "content": "You are a helpful assistant for system engineering."})

        if use_output_knowledge:
            for aik in self.knowledge:
                main_object["messages"].append({"role": "user", "content": aik.user_knowledge})
                main_object["messages"].append({"role": "assistant", "content": aik.assistant_knowledge})

        main_object["messages"].append({"role": "user", "content": text})

        try:
            response = requests.post(self.url_text, headers=headers, data=json.dumps(main_object))
            response.raise_for_status()
        except Exception as e:
            raise AIInterfaceException(str(e))

        try:
            answer_object = response.json()
            answer_array = answer_object["choices"]
            answer_text = answer_array[0]
            message_text = answer_text["message"]
            ai_text = message_text["content"]
        except Exception as e:
            raise AIInterfaceException(f"Failed to parse AI response: {e}")

        if use_output_knowledge:
            self.knowledge.append(AIKnowledge(text, ai_text))

        return ai_text

