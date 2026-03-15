from unittest.mock import Mock, patch

import pytest

from mem0.configs.llms.anthropic import AnthropicConfig
from mem0.llms.anthropic import AnthropicLLM

SIMPLE_TOOL = {
    "type": "function",
    "function": {
        "name": "my_tool",
        "description": "A tool",
        "parameters": {"type": "object", "properties": {}},
    },
}


@pytest.fixture
def mock_anthropic_client():
    with patch("mem0.llms.anthropic.anthropic") as mock_module:
        mock_client = Mock()
        mock_module.Anthropic.return_value = mock_client
        yield mock_client


@pytest.fixture
def llm(mock_anthropic_client):
    config = AnthropicConfig(model="claude-3-5-sonnet-20240620", api_key="test-key")
    return AnthropicLLM(config)


def _get_call_kwargs(mock_client):
    """Return the keyword arguments passed to messages.create."""
    call_kwargs = mock_client.messages.create.call_args
    return call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]


def _make_text_response(mock_client, text=""):
    """Configure mock_client to return a single text block response."""
    mock_block = Mock()
    mock_block.type = "text"
    mock_block.text = text
    mock_response = Mock()
    mock_response.content = [mock_block]
    mock_client.messages.create.return_value = mock_response


def test_generate_response_without_tools(llm, mock_anthropic_client):
    _make_text_response(mock_anthropic_client, "I'm doing well, thank you!")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]

    response = llm.generate_response(messages)

    assert response == "I'm doing well, thank you!"


def test_generate_response_with_tools_returns_tool_calls(llm, mock_anthropic_client):
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


def test_generate_response_with_tools_no_tool_use(llm, mock_anthropic_client):
    _make_text_response(mock_anthropic_client, "No tools needed.")
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

    response = llm.generate_response(messages, tools=tools)

    assert isinstance(response, dict)
    assert response["content"] == "No tools needed."
    assert response["tool_calls"] == []


def test_tool_format_conversion(llm, mock_anthropic_client):
    _make_text_response(mock_anthropic_client)
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

    llm.generate_response(messages, tools=tools)

    all_kwargs = _get_call_kwargs(mock_anthropic_client)
    assert len(all_kwargs["tools"]) == 1
    assert all_kwargs["tools"][0] == {
        "name": "my_tool",
        "description": "My tool description",
        "input_schema": {
            "type": "object",
            "properties": {"arg1": {"type": "string"}},
        },
    }
    assert all_kwargs["tool_choice"] == {"type": "auto"}


def test_tool_choice_required_maps_to_any(llm, mock_anthropic_client):
    _make_text_response(mock_anthropic_client)
    messages = [{"role": "user", "content": "test"}]

    llm.generate_response(messages, tools=[SIMPLE_TOOL], tool_choice="required")

    all_kwargs = _get_call_kwargs(mock_anthropic_client)
    assert all_kwargs["tool_choice"] == {"type": "any"}


def test_tool_choice_none_omits_param(llm, mock_anthropic_client):
    _make_text_response(mock_anthropic_client)
    messages = [{"role": "user", "content": "test"}]

    llm.generate_response(messages, tools=[SIMPLE_TOOL], tool_choice="none")

    all_kwargs = _get_call_kwargs(mock_anthropic_client)
    assert "tool_choice" not in all_kwargs


def test_tool_choice_specific_tool(llm, mock_anthropic_client):
    _make_text_response(mock_anthropic_client)
    messages = [{"role": "user", "content": "test"}]

    llm.generate_response(messages, tools=[SIMPLE_TOOL], tool_choice="my_tool")

    all_kwargs = _get_call_kwargs(mock_anthropic_client)
    assert all_kwargs["tool_choice"] == {"type": "tool", "name": "my_tool"}


def test_top_p_not_sent(mock_anthropic_client):
    """Anthropic rejects requests with both temperature and top_p; top_p is always dropped."""
    config = AnthropicConfig(model="claude-3-5-sonnet-20240620", api_key="test-key", top_p=0.9)
    llm = AnthropicLLM(config)
    _make_text_response(mock_anthropic_client, "response")
    messages = [{"role": "user", "content": "test"}]

    llm.generate_response(messages)

    all_kwargs = _get_call_kwargs(mock_anthropic_client)
    assert "top_p" not in all_kwargs


def test_response_format_json_appends_instruction(llm, mock_anthropic_client):
    _make_text_response(mock_anthropic_client, '{"result": "ok"}')
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Extract data"},
    ]

    llm.generate_response(messages, response_format={"type": "json_object"})

    all_kwargs = _get_call_kwargs(mock_anthropic_client)
    last_msg = all_kwargs["messages"][-1]
    assert last_msg["content"].endswith("\n\nYou must respond with valid JSON only.")
    # Original message should not be mutated
    assert messages[-1]["content"] == "Extract data"


def test_parse_response_raises_on_empty_content(llm, mock_anthropic_client):
    """_parse_response must raise ValueError rather than IndexError when content is empty."""
    mock_response = Mock()
    mock_response.content = []
    mock_anthropic_client.messages.create.return_value = mock_response
    messages = [{"role": "user", "content": "test"}]

    with pytest.raises(ValueError, match="Empty response from Anthropic API"):
        llm.generate_response(messages)


def test_convert_tools_raises_on_missing_function_key():
    """_convert_tools must raise ValueError when the 'function' key is absent."""
    with pytest.raises(ValueError, match="missing required key 'function'"):
        AnthropicLLM._convert_tools([{"type": "function"}])


def test_convert_tools_raises_on_missing_function_fields():
    """_convert_tools must raise ValueError when name/description/parameters are absent."""
    with pytest.raises(ValueError, match="missing required function key"):
        AnthropicLLM._convert_tools([{"type": "function", "function": {"name": "x"}}])


def test_response_format_suppressed_when_tools_provided(llm, mock_anthropic_client):
    """response_format JSON instruction must not be appended when tools are present."""
    _make_text_response(mock_anthropic_client)
    messages = [{"role": "user", "content": "Extract data"}]

    llm.generate_response(messages, response_format={"type": "json_object"}, tools=[SIMPLE_TOOL])

    all_kwargs = _get_call_kwargs(mock_anthropic_client)
    last_msg = all_kwargs["messages"][-1]
    assert "You must respond with valid JSON only." not in last_msg["content"]
