from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

class LLMClient(ABC):
    """
    Abstract interface for all LLM model providers.
    Allows seamless swapping between OpenAI, Llama, Anthropic, etc.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate a text response from the model.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system-level instructions.
            **kwargs: Model-specific arguments (temperature, max_tokens, top_p...).

        Returns:
            A plain text string response.
        """
        raise NotImplementedError
