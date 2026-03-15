import os
from typing import Dict, List, Optional, Union

try:
    import anthropic
except ImportError:
    raise ImportError("The 'anthropic' library is required. Please install it using 'pip install anthropic'.")

from mem0.configs.llms.anthropic import AnthropicConfig
from mem0.configs.llms.base import BaseLlmConfig
from mem0.llms.base import LLMBase


class AnthropicLLM(LLMBase):
    def __init__(self, config: Optional[Union[BaseLlmConfig, AnthropicConfig, Dict]] = None):
        if config is None:
            config = AnthropicConfig()
        elif isinstance(config, dict):
            config = AnthropicConfig(**config)
        elif isinstance(config, BaseLlmConfig) and not isinstance(config, AnthropicConfig):
            config = AnthropicConfig(
                model=config.model,
                temperature=config.temperature,
                api_key=config.api_key,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                top_k=config.top_k,
                enable_vision=config.enable_vision,
                vision_details=config.vision_details,
                http_client_proxies=config.http_client,
            )

        super().__init__(config)

        if not self.config.model:
            self.config.model = "claude-3-5-sonnet-20240620"

        api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ):
        """
        Generate a response based on the given messages using Anthropic.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            response_format (optional): When ``{"type": "json_object"}``, appends a JSON instruction
                to the last user message (suppressed when ``tools`` is provided). Defaults to None.
            tools (list, optional): List of tools that the model can call. Defaults to None.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".
            **kwargs: Additional Anthropic-specific parameters.

        Returns:
            str: The generated text response when no tools are provided.
            dict: A dict with keys ``content`` (str) and ``tool_calls`` (list) when tools are provided.
        """
        system_message = ""
        filtered_messages = []
        for message in messages:
            if message["role"] == "system":
                system_message = message["content"]
            else:
                filtered_messages.append(message)

        # Handle response_format for JSON output (same pattern as Ollama provider)
        if response_format and response_format.get("type") == "json_object" and not tools:
            if filtered_messages and filtered_messages[-1]["role"] == "user":
                filtered_messages[-1] = {
                    **filtered_messages[-1],
                    "content": filtered_messages[-1]["content"] + "\n\nYou must respond with valid JSON only.",
                }

        params = self._get_supported_params(**kwargs)
        # Anthropic rejects requests containing both temperature and top_p;
        # drop top_p since the config always provides a temperature default.
        if "top_p" in params:
            params.pop("top_p")
        params.update(
            {
                "model": self.config.model,
                "messages": filtered_messages,
                "system": system_message,
            }
        )

        if tools:
            params["tools"] = self._convert_tools(tools)
            mapped = self._map_tool_choice(tool_choice)
            if mapped is not None:
                params["tool_choice"] = mapped

        response = self.client.messages.create(**params)
        return self._parse_response(response, tools)

    @staticmethod
    def _map_tool_choice(tool_choice):
        """Map OpenAI-style tool_choice values to Anthropic format.

        Returns ``None`` for ``"none"`` so the caller can omit the parameter.
        """
        if tool_choice == "auto":
            return {"type": "auto"}
        if tool_choice == "required":
            return {"type": "any"}
        if tool_choice == "none":
            return None
        # Specific tool name
        return {"type": "tool", "name": tool_choice}

    @staticmethod
    def _convert_tools(tools):
        """Convert OpenAI-format tools to Anthropic format.

        Raises:
            ValueError: If a tool entry is missing required keys (``function``,
                ``name``, ``description``, or ``parameters``).
        """
        converted = []
        for i, tool in enumerate(tools):
            if "function" not in tool:
                raise ValueError(f"Tool at index {i} is missing required key 'function'")
            func = tool["function"]
            for key in ("name", "description", "parameters"):
                if key not in func:
                    raise ValueError(f"Tool at index {i} is missing required function key '{key}'")
            converted.append(
                {
                    "name": func["name"],
                    "description": func["description"],
                    "input_schema": func["parameters"],
                }
            )
        return converted

    @staticmethod
    def _parse_response(response, tools):
        """Parse Anthropic response, extracting tool calls when tools were provided."""
        if not response.content:
            raise ValueError("Empty response from Anthropic API")

        if tools:
            content = ""
            tool_calls = []
            for block in response.content:
                if block.type == "text":
                    content = block.text
                elif block.type == "tool_use":
                    tool_calls.append(
                        {
                            "name": block.name,
                            "arguments": block.input,
                        }
                    )
            return {"content": content, "tool_calls": tool_calls}
        return response.content[0].text
