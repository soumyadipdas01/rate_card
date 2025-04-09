import pandas as pd
from pandasai import SmartDataframe
from pandasai.exceptions import MaliciousQueryError, NoResultFoundError
import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextGenerationModel
from vertexai.preview.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models
from google.cloud import aiplatform
from google.oauth2 import service_account
from pandasai.llm import OpenAI
from typing import Any, Optional, abstractmethod, Dict
from pandasai.prompts.base import BasePrompt
from pandasai.helpers.memory import Memory
from pandasai.llm.base import LLM
from vertexai.preview.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models
safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
}

class BaseGeminiAI(LLM):
    temperature: float = 0
    top_p: float = 0.95
    top_k: int = 40
    max_output_tokens: int = 8192
    stop: Optional[str] = None
    client: Any
    _is_chat_model: bool

    def _set_params(self, **kwargs):
        valid_params = [
            "model",
            "temperature",
            "max_tokens",
            "top_p",
        ]
        for key, value in kwargs.items():
            if key in valid_params:
                setattr(self, key, value)

    @property
    def _default_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }

        return params


    def chat_completion(self, value: str) -> str:
        messages = []

        messages.append(
            {"role": "user", "parts": [{"text" : value}]}
        )
        

        responses = self.client.generate_content(messages,generation_config={**self._default_params}, safety_settings=safety_settings,stream=True)
        r = ""
        for response in responses:
            r = r + response.text
        return r


    def call(self, instruction: BasePrompt, context: Any):
        self.last_prompt = instruction.to_string()
        return (
            self.chat_completion(self.last_prompt)
            #if self._is_chat_model
            #else self.completion(self.last_prompt)
        )
    

class geminiAI(BaseGeminiAI):

    model: str = "gemini-1.5-pro-002"
    creds_llm = service_account.Credentials.from_service_account_file("infy_auto.json")
    vertexai.init(project="upheld-caldron-411606", location="us-central1", credentials=creds_llm)

    def __init__(self):
        self.client = GenerativeModel(self.model)
    
    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API"""
        return {
            **super()._default_params,
        }

    @property
    def type(self) -> str:
        return "googlegemini"