import os
from abc import ABC, abstractmethod

import torch
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import T5ForConditionalGeneration, T5Tokenizer


def _build_openai_client(api_key=None, base_url=None):
    client_kwargs = {"api_key": api_key or os.environ.get("OPENAI_API_KEY")}
    resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL")
    if resolved_base_url:
        client_kwargs["base_url"] = resolved_base_url
    return OpenAI(**client_kwargs)


class BaseQAModel(ABC):
    @abstractmethod
    def answer_question(self, context, question):
        pass


class GPT3QAModel(BaseQAModel):
    def __init__(self, model="text-davinci-003", api_key=None, base_url=None):
        """
        Initializes the GPT-3 QA model with the specified model version.
        """
        self.model = model
        self.client = _build_openai_client(api_key=api_key, base_url=base_url)

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, max_tokens=150, stop_sequence=None):
        """
        Answers a question given the provided context using the GPT-3 model.
        """
        response = self.client.completions.create(
            prompt=f"Using the following information {context}. Answer the following question in less than 5-7 words, if possible: {question}",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=self.model,
        )
        return response.choices[0].text.strip()


class GPT3TurboQAModel(BaseQAModel):
    def __init__(self, model="gpt-4o-mini", api_key=None, base_url=None):
        """
        Initializes the chat completion QA model with the specified model version.
        """
        self.model = model
        self.client = _build_openai_client(api_key=api_key, base_url=base_url)

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _attempt_answer_question(
        self, context, question, max_tokens=150, stop_sequence=None
    ):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are Question Answering Portal"},
                {
                    "role": "user",
                    "content": (
                        f"Please provide a clear and brief response to the following question: {question}. "
                        "Respond with 'Yes' or 'No' only when the question is a general yes-no question, "
                        f"starting with 'Do', 'Did', 'Does', 'Was' or 'Is'. Using the following information {context}."
                    ),
                },
            ],
            temperature=0,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    def answer_question(self, context, question, max_tokens=150, stop_sequence=None):
        return self._attempt_answer_question(
            context, question, max_tokens=max_tokens, stop_sequence=stop_sequence
        )


class GPT4QAModel(BaseQAModel):
    def __init__(self, model="gpt-4", api_key=None, base_url=None):
        """
        Initializes the GPT-4 QA model with the specified model version.
        """
        self.model = model
        self.client = _build_openai_client(api_key=api_key, base_url=base_url)

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _attempt_answer_question(
        self, context, question, max_tokens=150, stop_sequence=None
    ):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are Question Answering Portal"},
                {
                    "role": "user",
                    "content": f"Given Context: {context} Give the best full answer amongst the option to question {question}",
                },
            ],
            temperature=0,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    def answer_question(self, context, question, max_tokens=150, stop_sequence=None):
        return self._attempt_answer_question(
            context, question, max_tokens=max_tokens, stop_sequence=stop_sequence
        )


class UnifiedQAModel(BaseQAModel):
    def __init__(self, model_name="allenai/unifiedqa-v2-t5-3b-1363200"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(
            self.device
        )
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def run_model(self, input_string, **generator_args):
        input_ids = self.tokenizer.encode(input_string, return_tensors="pt").to(
            self.device
        )
        res = self.model.generate(input_ids, **generator_args)
        return self.tokenizer.batch_decode(res, skip_special_tokens=True)

    def answer_question(self, context, question):
        input_string = question + " \\n " + context
        output = self.run_model(input_string)
        return output[0]

