from unittest.mock import Mock, patch

import pytest

from mem0.configs.llms.anthropic import AnthropicConfig
from mem0.llms.anthropic import AnthropicLLM


@pytest.fixture
def mock_anthropic_client():
    with patch("mem0.llms.anthropic.anthropic") as mock_module:
        mock_client = Mock()
        mock_module.Anthropic.return_value = mock_client
        yield mock_client


def test_generate_response_without_tools(mock_anthropic_client):
    config = AnthropicConfig(model="claude-3-5-sonnet-20240620", api_key="test-key")
    llm = AnthropicLLM(config)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]

    mock_text_block = Mock()
    mock_text_block.type = "text"
    mock_text_block.text = "I'm doing well, thank you!"

    mock_response = Mock()
    mock_response.content = [mock_text_block]
    mock_anthropic_client.messages.create.return_value = mock_response

    response = llm.generate_response(messages)

    assert response == "I'm doing well, thank you!"


def test_generate_response_with_tools_returns_tool_calls(mock_anthropic_client):
    config = AnthropicConfig(model="claude-3-5-sonnet-20240620", api_key="test-key")
    llm = AnthropicLLM(config)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Extract entities from: Alice likes pizza"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "extract_entities",
                "description": "Extract entities from text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entities": {
                            "type": "array",
                            "items": {"type": "object"},
                        }
                    },
                },
            },
        }
    ]

    mock_text_block = Mock()
    mock_text_block.type = "text"
    mock_text_block.text = "I found some entities."

    mock_tool_block = Mock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.name = "extract_entities"
    mock_tool_block.input = {"entities": [{"entity": "Alice", "entity_type": "person"}]}

    mock_response = Mock()
    mock_response.content = [mock_text_block, mock_tool_block]
    mock_anthropic_client.messages.create.return_value = mock_response

    response = llm.generate_response(messages, tools=tools)

    assert isinstance(response, dict)
    assert response["content"] == "I found some entities."
    assert len(response["tool_calls"]) == 1
    assert response["tool_calls"][0]["name"] == "extract_entities"
    assert response["tool_calls"][0]["arguments"] == {
        "entities": [{"entity": "Alice", "entity_type": "person"}]
    }


def test_generate_response_with_tools_no_tool_use(mock_anthropic_client):
    config = AnthropicConfig(model="claude-3-5-sonnet-20240620", api_key="test-key")
    llm = AnthropicLLM(config)
    messages = [{"role": "user", "content": "Hello"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "some_tool",
                "description": "A tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    mock_text_block = Mock()
    mock_text_block.type = "text"
    mock_text_block.text = "No tools needed."

    mock_response = Mock()
    mock_response.content = [mock_text_block]
    mock_anthropic_client.messages.create.return_value = mock_response

    response = llm.generate_response(messages, tools=tools)

    assert isinstance(response, dict)
    assert response["content"] == "No tools needed."
    assert response["tool_calls"] == []


def test_tool_format_conversion(mock_anthropic_client):
    config = AnthropicConfig(model="claude-3-5-sonnet-20240620", api_key="test-key")
    llm = AnthropicLLM(config)
    messages = [{"role": "user", "content": "test"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "my_tool",
                "description": "My tool description",
                "parameters": {
                    "type": "object",
                    "properties": {"arg1": {"type": "string"}},
                },
            },
        }
    ]

    mock_text_block = Mock()
    mock_text_block.type = "text"
    mock_text_block.text = ""

    mock_response = Mock()
    mock_response.content = [mock_text_block]
    mock_anthropic_client.messages.create.return_value = mock_response

    llm.generate_response(messages, tools=tools)

    call_kwargs = mock_anthropic_client.messages.create.call_args
    passed_tools = call_kwargs.kwargs.get("tools") or call_kwargs[1].get("tools")

    assert len(passed_tools) == 1
    assert passed_tools[0] == {
        "name": "my_tool",
        "description": "My tool description",
        "input_schema": {
            "type": "object",
            "properties": {"arg1": {"type": "string"}},
        },
    }

    passed_tool_choice = call_kwargs.kwargs.get("tool_choice") or call_kwargs[1].get("tool_choice")
    assert passed_tool_choice == {"type": "auto"}


def test_tool_choice_required_maps_to_any(mock_anthropic_client):
    config = AnthropicConfig(model="claude-3-5-sonnet-20240620", api_key="test-key")
    llm = AnthropicLLM(config)
    messages = [{"role": "user", "content": "test"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "my_tool",
                "description": "A tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    mock_text_block = Mock()
    mock_text_block.type = "text"
    mock_text_block.text = ""
    mock_response = Mock()
    mock_response.content = [mock_text_block]
    mock_anthropic_client.messages.create.return_value = mock_response

    llm.generate_response(messages, tools=tools, tool_choice="required")

    call_kwargs = mock_anthropic_client.messages.create.call_args
    passed_tool_choice = call_kwargs.kwargs.get("tool_choice") or call_kwargs[1].get("tool_choice")
    assert passed_tool_choice == {"type": "any"}


def test_tool_choice_none_omits_param(mock_anthropic_client):
    config = AnthropicConfig(model="claude-3-5-sonnet-20240620", api_key="test-key")
    llm = AnthropicLLM(config)
    messages = [{"role": "user", "content": "test"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "my_tool",
                "description": "A tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    mock_text_block = Mock()
    mock_text_block.type = "text"
    mock_text_block.text = ""
    mock_response = Mock()
    mock_response.content = [mock_text_block]
    mock_anthropic_client.messages.create.return_value = mock_response

    llm.generate_response(messages, tools=tools, tool_choice="none")

    call_kwargs = mock_anthropic_client.messages.create.call_args
    all_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
    assert "tool_choice" not in all_kwargs


def test_tool_choice_specific_tool(mock_anthropic_client):
    config = AnthropicConfig(model="claude-3-5-sonnet-20240620", api_key="test-key")
    llm = AnthropicLLM(config)
    messages = [{"role": "user", "content": "test"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "my_tool",
                "description": "A tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    mock_text_block = Mock()
    mock_text_block.type = "text"
    mock_text_block.text = ""
    mock_response = Mock()
    mock_response.content = [mock_text_block]
    mock_anthropic_client.messages.create.return_value = mock_response

    llm.generate_response(messages, tools=tools, tool_choice="my_tool")

    call_kwargs = mock_anthropic_client.messages.create.call_args
    passed_tool_choice = call_kwargs.kwargs.get("tool_choice") or call_kwargs[1].get("tool_choice")
    assert passed_tool_choice == {"type": "tool", "name": "my_tool"}


def test_top_p_not_sent(mock_anthropic_client):
    config = AnthropicConfig(model="claude-3-5-sonnet-20240620", api_key="test-key", top_p=0.9)
    llm = AnthropicLLM(config)
    messages = [{"role": "user", "content": "test"}]

    mock_text_block = Mock()
    mock_text_block.type = "text"
    mock_text_block.text = "response"
    mock_response = Mock()
    mock_response.content = [mock_text_block]
    mock_anthropic_client.messages.create.return_value = mock_response

    llm.generate_response(messages)

    call_kwargs = mock_anthropic_client.messages.create.call_args
    all_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
    assert "top_p" not in all_kwargs


def test_response_format_json_appends_instruction(mock_anthropic_client):
    config = AnthropicConfig(model="claude-3-5-sonnet-20240620", api_key="test-key")
    llm = AnthropicLLM(config)
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Extract data"},
    ]

    mock_text_block = Mock()
    mock_text_block.type = "text"
    mock_text_block.text = '{"result": "ok"}'
    mock_response = Mock()
    mock_response.content = [mock_text_block]
    mock_anthropic_client.messages.create.return_value = mock_response

    llm.generate_response(messages, response_format={"type": "json_object"})

    call_kwargs = mock_anthropic_client.messages.create.call_args
    all_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
    passed_messages = all_kwargs["messages"]
    last_msg = passed_messages[-1]
    assert last_msg["content"].endswith("\n\nYou must respond with valid JSON only.")
    # Original message should not be mutated
    assert messages[-1]["content"] == "Extract data"
