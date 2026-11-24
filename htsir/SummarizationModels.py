import logging
import os
from abc import ABC, abstractmethod

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BaseSummarizationModel(ABC):
    @abstractmethod
    def summarize(self, context, max_tokens=150):
        pass


class GPT3TurboSummarizationModel(BaseSummarizationModel):
    def __init__(self, model="gpt-4o-mini", api_key=None, base_url=None):
        self.model = model
        client_kwargs = {"api_key": api_key or os.environ.get("OPENAI_API_KEY")}
        resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url
        self.client = OpenAI(**client_kwargs)

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": f"Write a summary of the following, including as many key details as possible: {context}:",
                },
            ],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


class GPT3SummarizationModel(BaseSummarizationModel):
    def __init__(self, model="text-davinci-003", api_key=None, base_url=None):
        self.model = model
        client_kwargs = {"api_key": api_key or os.environ.get("OPENAI_API_KEY")}
        resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url
        self.client = OpenAI(**client_kwargs)

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=200, stop_sequence=None):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": f"Write a summary of the following, including as many key details as possible: {context}:",
                },
            ],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

