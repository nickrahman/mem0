/**
 * Tests for AnthropicLLM tool support (issue #3711).
 *
 * Verifies that AnthropicLLM correctly:
 *   - Returns a plain string when no tools are provided
 *   - Converts OpenAI-format tools to Anthropic format
 *   - Parses tool_use response blocks into LLMResponse with toolCalls
 *   - Returns empty toolCalls when tools are provided but LLM doesn't use them
 */

import { AnthropicLLM } from "../src/llms/anthropic";

// Mock the Anthropic SDK
const mockCreate = jest.fn();
jest.mock("@anthropic-ai/sdk", () => {
  return {
    __esModule: true,
    default: jest.fn().mockImplementation(() => ({
      messages: { create: mockCreate },
    })),
  };
});

const TOOLS = [
  {
    type: "function",
    function: {
      name: "extract_entities",
      description: "Extract entities from text",
      parameters: {
        type: "object",
        properties: {
          entities: { type: "array", items: { type: "object" } },
        },
      },
    },
  },
];

const MESSAGES = [
  { role: "system" as const, content: "You are helpful." },
  { role: "user" as const, content: "Extract entities from: Alice likes pizza" },
];

beforeEach(() => {
  jest.clearAllMocks();
});

describe("AnthropicLLM", () => {
  it("returns a plain string when no tools are provided", async () => {
    mockCreate.mockResolvedValueOnce({
      content: [{ type: "text", text: "Hello there!" }],
    });

    const llm = new AnthropicLLM({
      apiKey: "test-key",
      model: "claude-3-sonnet-20240229",
    });
    const result = await llm.generateResponse(MESSAGES);

    expect(result).toBe("Hello there!");
  });

  it("returns LLMResponse with toolCalls when tools are used", async () => {
    mockCreate.mockResolvedValueOnce({
      content: [
        { type: "text", text: "Found entities." },
        {
          type: "tool_use",
          name: "extract_entities",
          input: {
            entities: [{ entity: "Alice", entity_type: "person" }],
          },
        },
      ],
    });

    const llm = new AnthropicLLM({
      apiKey: "test-key",
      model: "claude-3-sonnet-20240229",
    });
    const result = await llm.generateResponse(MESSAGES, undefined, TOOLS);

    expect(typeof result).toBe("object");
    expect(result).toHaveProperty("content", "Found entities.");
    expect(result).toHaveProperty("toolCalls");

    const toolCalls = (result as any).toolCalls;
    expect(toolCalls).toHaveLength(1);
    expect(toolCalls[0].name).toBe("extract_entities");
    expect(JSON.parse(toolCalls[0].arguments)).toEqual({
      entities: [{ entity: "Alice", entity_type: "person" }],
    });
  });

  it("returns empty toolCalls when tools provided but not used", async () => {
    mockCreate.mockResolvedValueOnce({
      content: [{ type: "text", text: "No tools needed." }],
    });

    const llm = new AnthropicLLM({
      apiKey: "test-key",
      model: "claude-3-sonnet-20240229",
    });
    const result = await llm.generateResponse(MESSAGES, undefined, TOOLS);

    expect(typeof result).toBe("object");
    expect(result).toHaveProperty("content", "No tools needed.");
    expect((result as any).toolCalls).toEqual([]);
  });

  it("converts OpenAI-format tools to Anthropic format", async () => {
    mockCreate.mockResolvedValueOnce({
      content: [{ type: "text", text: "" }],
    });

    const llm = new AnthropicLLM({
      apiKey: "test-key",
      model: "claude-3-sonnet-20240229",
    });
    await llm.generateResponse(MESSAGES, undefined, TOOLS);

    const callArgs = mockCreate.mock.calls[0][0];
    expect(callArgs.tools).toEqual([
      {
        name: "extract_entities",
        description: "Extract entities from text",
        input_schema: {
          type: "object",
          properties: {
            entities: { type: "array", items: { type: "object" } },
          },
        },
      },
    ]);
    expect(callArgs.tool_choice).toEqual({ type: "auto" });
  });

  it("maps tool_choice required to any", async () => {
    mockCreate.mockResolvedValueOnce({
      content: [{ type: "text", text: "" }],
    });

    const llm = new AnthropicLLM({
      apiKey: "test-key",
      model: "claude-3-sonnet-20240229",
    });
    await llm.generateResponse(MESSAGES, undefined, TOOLS, "required");

    const callArgs = mockCreate.mock.calls[0][0];
    expect(callArgs.tool_choice).toEqual({ type: "any" });
  });

  it("omits tool_choice when none", async () => {
    mockCreate.mockResolvedValueOnce({
      content: [{ type: "text", text: "" }],
    });

    const llm = new AnthropicLLM({
      apiKey: "test-key",
      model: "claude-3-sonnet-20240229",
    });
    await llm.generateResponse(MESSAGES, undefined, TOOLS, "none");

    const callArgs = mockCreate.mock.calls[0][0];
    expect(callArgs.tool_choice).toBeUndefined();
  });

  it("maps tool_choice to specific tool name", async () => {
    mockCreate.mockResolvedValueOnce({
      content: [{ type: "text", text: "" }],
    });

    const llm = new AnthropicLLM({
      apiKey: "test-key",
      model: "claude-3-sonnet-20240229",
    });
    await llm.generateResponse(MESSAGES, undefined, TOOLS, "extract_entities");

    const callArgs = mockCreate.mock.calls[0][0];
    expect(callArgs.tool_choice).toEqual({ type: "tool", name: "extract_entities" });
  });

  it("appends JSON instruction when response_format is json_object", async () => {
    mockCreate.mockResolvedValueOnce({
      content: [{ type: "text", text: '{"result": "ok"}' }],
    });

    const llm = new AnthropicLLM({
      apiKey: "test-key",
      model: "claude-3-sonnet-20240229",
    });
    await llm.generateResponse(MESSAGES, { type: "json_object" });

    const callArgs = mockCreate.mock.calls[0][0];
    const lastMsg = callArgs.messages[callArgs.messages.length - 1];
    expect(lastMsg.content).toContain("You must respond with valid JSON only.");
    // Original messages should not be mutated
    expect(MESSAGES[1].content).toBe("Extract entities from: Alice likes pizza");
  });

  it("does not append JSON instruction when tools are provided", async () => {
    mockCreate.mockResolvedValueOnce({
      content: [{ type: "text", text: "" }],
    });

    const llm = new AnthropicLLM({
      apiKey: "test-key",
      model: "claude-3-sonnet-20240229",
    });
    await llm.generateResponse(MESSAGES, { type: "json_object" }, TOOLS);

    const callArgs = mockCreate.mock.calls[0][0];
    const lastMsg = callArgs.messages[callArgs.messages.length - 1];
    expect(lastMsg.content).not.toContain("You must respond with valid JSON only.");
  });
});
