import os
from typing import Optional, Dict, Any

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Please install openai>=1.0.0: pip install openai")

from clients.llm_client import LLMClient


class OpenAIClient(LLMClient):
    """Simple OpenAI-backed implementation of LLMClient using OpenAI v1.0+ API.

    This wrapper keeps the surface small: call `generate(prompt, system_prompt, **kwargs)`
    to get a text response. It reads the API key from the environment variable
    `OPENAI_API_KEY` if one is not provided.

    Notes:
    - `kwargs` are forwarded to the OpenAI chat completions call and can
      include model-specific parameters such as `temperature`, `max_tokens`.
    - The default model may be overridden when constructing the client.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.model = model
        # Initialize OpenAI client with API key
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            # OpenAI client will automatically use OPENAI_API_KEY env var
            self.client = OpenAI()

    def generate(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> str:
        """Generate text from the OpenAI chat completion API.

        Args:
            prompt: The user-level prompt (string).
            system_prompt: Optional system instructions to include as the first message.
            **kwargs: Extra parameters forwarded to OpenAI (temperature, max_tokens, etc.)

        Returns:
            The assistant's text response as a string.

        Raises:
            RuntimeError: if the OpenAI call fails or returns no content.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs,
            )
        except Exception as exc:
            # Wrap exceptions to avoid leaking library-specific types to callers
            raise RuntimeError(f"OpenAI request failed: {exc}") from exc

        # Extract text safely
        try:
            content = response.choices[0].message.content
        except Exception as exc:
            raise RuntimeError(f"Unexpected OpenAI response shape: {exc}") from exc

        if not content:
            raise RuntimeError("OpenAI returned an empty response")

        return content
