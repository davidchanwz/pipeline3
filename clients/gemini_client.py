import os
from typing import Optional, Dict, Any

try:
    import google.generativeai as genai
except ImportError:
    raise ImportError(
        "Please install google-generativeai: pip install google-generativeai"
    )

from clients.llm_client import LLMClient


class GeminiClient(LLMClient):
    """Simple Gemini-backed implementation of LLMClient using Google AI Studio API.

    This wrapper keeps the surface small: call `generate(prompt, system_prompt, **kwargs)`
    to get a text response. It reads the API key from the environment variable
    `GOOGLE_API_KEY` if one is not provided.

    Notes:
    - `kwargs` are forwarded to the Gemini generate_content call and can
      include model-specific parameters such as `temperature`, `max_output_tokens`.
    - The default model may be overridden when constructing the client.
    """

    def __init__(
        self, model: str = "gemini-2.0-flash-lite", api_key: Optional[str] = None
    ):
        self.model_name = model
        self.model = model  # Keep model name as string for compatibility

        # Configure API key
        if api_key:
            genai.configure(api_key=api_key)
        else:
            # Check for API key in environment
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "GOOGLE_API_KEY environment variable not set and no api_key provided"
                )
            genai.configure(api_key=api_key)

        # Initialize the generative model object
        self.generative_model = genai.GenerativeModel(self.model_name)

    def generate(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> str:
        """Generate text from the Gemini API.

        Args:
            prompt: The user-level prompt (string).
            system_prompt: Optional system instructions to include.
            **kwargs: Extra parameters forwarded to Gemini (temperature, max_output_tokens, etc.)

        Returns:
            The model's text response as a string.

        Raises:
            RuntimeError: if the Gemini call fails or returns no content.
        """
        try:
            # Combine system prompt and user prompt if system prompt exists
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"

            # Set up generation config from kwargs
            generation_config = {}
            if "temperature" in kwargs:
                generation_config["temperature"] = kwargs["temperature"]
            if "max_tokens" in kwargs:
                generation_config["max_output_tokens"] = kwargs["max_tokens"]
            if "max_output_tokens" in kwargs:
                generation_config["max_output_tokens"] = kwargs["max_output_tokens"]
            if "top_p" in kwargs:
                generation_config["top_p"] = kwargs["top_p"]
            if "top_k" in kwargs:
                generation_config["top_k"] = kwargs["top_k"]

            # Generate content
            response = self.generative_model.generate_content(
                full_prompt,
                generation_config=generation_config if generation_config else None,
            )

        except Exception as exc:
            # Wrap exceptions to avoid leaking library-specific types to callers
            raise RuntimeError(f"Gemini request failed: {exc}") from exc

        # Extract text safely
        try:
            content = response.text
        except Exception as exc:
            raise RuntimeError(f"Unexpected Gemini response shape: {exc}") from exc

        if not content:
            raise RuntimeError("Gemini returned an empty response")

        return content
