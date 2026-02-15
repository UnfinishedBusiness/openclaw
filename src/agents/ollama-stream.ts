import type { StreamFn } from "@mariozechner/pi-agent-core";
import type {
  AssistantMessage,
  StopReason,
  TextContent,
  ToolCall,
  Tool,
  Usage,
} from "@mariozechner/pi-ai";
import { createAssistantMessageEventStream } from "@mariozechner/pi-ai";
import { randomUUID } from "node:crypto";
import { createSubsystemLogger } from "src/logging/subsystem.ts";

// Dynamic import for Ollama JS (cloud)
let OllamaJs: typeof import("ollama").Ollama | undefined;
try {
  // @ts-ignore
  OllamaJs = require("ollama").Ollama;
} catch {
  try {
    // @ts-ignore
    OllamaJs = (await import("ollama")).Ollama;
  } catch {}
}

export const OLLAMA_NATIVE_BASE_URL = "http://127.0.0.1:11434";
const log = createSubsystemLogger("src/agents/ollama-stream");

// ── Types ────────────────────────────────────────────────────────────────
interface OllamaChatMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  images?: string[];
  tool_calls?: OllamaToolCall[];
  tool_name?: string;
}
interface OllamaTool {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}
interface OllamaToolCall {
  function: {
    name: string;
    arguments: Record<string, unknown>;
  };
}

// Request body for Ollama chat API (local)
interface OllamaChatRequest {
  model: string;
  messages: OllamaChatMessage[];
  stream: boolean;
  tools?: OllamaTool[];
  options?: Record<string, unknown>;
}
interface OllamaChatResponse {
  model: string;
  created_at: string;
  message: {
    role: "assistant";
    content: string;
    tool_calls?: OllamaToolCall[];
  };
  done: boolean;
  done_reason?: string;
  total_duration?: number;
  load_duration?: number;
  prompt_eval_count?: number;
  prompt_eval_duration?: number;
  eval_count?: number;
  eval_duration?: number;
}

// ── Helpers ───────────────────────────────────────────────────────────────
function extractTextContent(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }
  if (!Array.isArray(content)) {
    return "";
  }
  return content
    .filter((part) => part.type === "text")
    .map((part) => part.text)
    .join("");
}
function extractOllamaImages(content: unknown): string[] {
  if (!Array.isArray(content)) {
    return [];
  }
  return content.filter((part) => part.type === "image").map((part) => part.data);
}
function extractToolCalls(content: unknown): OllamaToolCall[] {
  if (!Array.isArray(content)) {
    return [];
  }
  const result: OllamaToolCall[] = [];
  for (const part of content) {
    if (part.type === "toolCall") {
      result.push({ function: { name: part.name, arguments: part.arguments } });
    } else if (part.type === "tool_use") {
      result.push({ function: { name: part.name, arguments: part.input } });
    }
  }
  return result;
}
function convertToOllamaMessages(
  messages: Array<{ role: string; content: unknown }>,
  system?: string,
): OllamaChatMessage[] {
  const result: OllamaChatMessage[] = [];
  if (system) {
    result.push({ role: "system", content: system });
  }
  for (const msg of messages) {
    const { role } = msg;
    if (role === "user") {
      const text = extractTextContent(msg.content);
      const images = extractOllamaImages(msg.content);
      result.push({
        role: "user",
        content: text,
        ...(images.length > 0 ? { images } : {}),
      });
    } else if (role === "assistant") {
      const text = extractTextContent(msg.content);
      const toolCalls = extractToolCalls(msg.content);
      result.push({
        role: "assistant",
        content: text,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
    } else if (role === "tool" || role === "toolResult") {
      const text = extractTextContent(msg.content);
      const toolName =
        typeof (msg as any).toolName === "string" ? (msg as any).toolName : undefined;
      result.push({
        role: "tool",
        content: text,
        ...(toolName ? { tool_name: toolName } : {}),
      });
    }
  }
  return result;
}
function extractOllamaTools(tools: Tool[] | undefined): OllamaTool[] {
  if (!tools || !Array.isArray(tools)) {
    return [];
  }
  const result: OllamaTool[] = [];
  for (const tool of tools) {
    if (typeof tool.name !== "string" || !tool.name) {
      continue;
    }
    result.push({
      type: "function",
      function: {
        name: tool.name,
        description: typeof tool.description === "string" ? tool.description : "",
        parameters: (tool.parameters ?? {}) as Record<string, unknown>,
      },
    });
  }
  return result;
}
export function buildAssistantMessage(
  response: OllamaChatResponse,
  modelInfo: { api: string; provider: string; id: string },
): AssistantMessage {
  const content: (TextContent | ToolCall)[] = [];
  if (response.message.content) {
    content.push({ type: "text", text: response.message.content });
  }
  const toolCalls = response.message.tool_calls;
  if (toolCalls && toolCalls.length > 0) {
    for (const tc of toolCalls) {
      content.push({
        type: "toolCall",
        id: `ollama_call_${randomUUID()}`,
        name: tc.function.name,
        arguments: tc.function.arguments,
      });
    }
  }
  const hasToolCalls = toolCalls && toolCalls.length > 0;
  const stopReason: StopReason = hasToolCalls ? "toolUse" : "stop";
  const usage: Usage = {
    input: response.prompt_eval_count ?? 0,
    output: response.eval_count ?? 0,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens: (response.prompt_eval_count ?? 0) + (response.eval_count ?? 0),
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  };
  return {
    role: "assistant",
    content,
    stopReason,
    api: modelInfo.api,
    provider: modelInfo.provider,
    model: modelInfo.id,
    usage,
    timestamp: Date.now(),
  };
}
export async function* parseNdjsonStream(
  reader: ReadableStreamDefaultReader<Uint8Array>,
): AsyncGenerator<OllamaChatResponse> {
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) {
        continue;
      }
      try {
        yield JSON.parse(trimmed) as OllamaChatResponse;
      } catch {
        // skip malformed
      }
    }
  }
  if (buffer.trim()) {
    try {
      yield JSON.parse(buffer.trim()) as OllamaChatResponse;
    } catch {}
  }
}
function resolveOllamaChatUrl(baseUrl: string): string {
  const trimmed = baseUrl.trim().replace(/\/+$/, "");
  const normalizedBase = trimmed.replace(/\/v1$/i, "");
  const apiBase = normalizedBase || OLLAMA_NATIVE_BASE_URL;
  return `${apiBase}/api/chat`;
}

// ── Main StreamFn factory ────────────────────────────────────────────────
export function createOllamaStreamFn(baseUrl: string): StreamFn {
  return (model, context, options) => {
    const isLocal = /^https?:\/\/(127\.0\.0\.1|localhost)(:[0-9]+)?(\/|$)/.test(baseUrl);
    const chatUrl = resolveOllamaChatUrl(baseUrl);
    const stream = createAssistantMessageEventStream();
    const run = async () => {
      try {
        const ollamaMessages = convertToOllamaMessages(
          context.messages ?? [],
          context.systemPrompt,
        );
        const ollamaTools = extractOllamaTools(context.tools);
        const ollamaOptions: Record<string, unknown> = { num_ctx: model.contextWindow ?? 65536 };
        if (typeof options?.temperature === "number") {
          ollamaOptions.temperature = options.temperature;
        }
        if (typeof options?.maxTokens === "number") {
          ollamaOptions.num_predict = options.maxTokens;
        }
        if (!isLocal && OllamaJs) {
          // Use Ollama JS library for cloud
          const client = new OllamaJs({ host: baseUrl });
          let accumulatedContent = "";
          const accumulatedToolCalls: OllamaToolCall[] = [];
          let finalResponse: OllamaChatResponse | undefined;
          const chatStream = await client.chat({
            model: model.id,
            messages: ollamaMessages,
            stream: true,
            tools: ollamaTools.length > 0 ? ollamaTools : undefined,
            options: ollamaOptions,
          });
          for await (const chunk of chatStream) {
            if (chunk.message?.content) {
              accumulatedContent += chunk.message.content;
            }
            if (chunk.message?.tool_calls) {
              accumulatedToolCalls.push(...chunk.message.tool_calls);
            }
            if (chunk.done) {
              finalResponse = {
                ...chunk,
                created_at:
                  chunk.created_at instanceof Date
                    ? chunk.created_at.toISOString()
                    : chunk.created_at,
              } as OllamaChatResponse;
              break;
            }
          }
          if (!finalResponse) {
            throw new Error("Ollama API stream ended without a final response");
          }
          finalResponse.message.content = accumulatedContent;
          if (accumulatedToolCalls.length > 0) {
            finalResponse.message.tool_calls = accumulatedToolCalls;
          }
          const assistantMessage = buildAssistantMessage(finalResponse, {
            api: model.api,
            provider: model.provider,
            id: model.id,
          });
          const reason: Extract<StopReason, "stop" | "length" | "toolUse"> =
            assistantMessage.stopReason === "toolUse" ? "toolUse" : "stop";
          stream.push({ type: "done", reason, message: assistantMessage });
        } else {
          // Use HTTP/NDJSON for local
          const body: OllamaChatRequest = {
            model: model.id,
            messages: ollamaMessages,
            stream: true,
            ...(ollamaTools.length > 0 ? { tools: ollamaTools } : {}),
            options: ollamaOptions,
          };
          const headers: Record<string, string> = {
            "Content-Type": "application/json",
            ...options?.headers,
          };
          if (options?.apiKey) {
            headers.Authorization = `Bearer ${options.apiKey}`;
          }
          const response = await fetch(chatUrl, {
            method: "POST",
            headers,
            body: JSON.stringify(body),
            signal: options?.signal,
          });
          if (!response.ok) {
            const errorText = await response.text().catch(() => "unknown error");
            throw new Error(`Ollama API error ${response.status}: ${errorText}`);
          }
          if (!response.body) {
            throw new Error("Ollama API returned empty response body");
          }
          const reader = response.body.getReader();
          let accumulatedContent = "";
          const accumulatedToolCalls: OllamaToolCall[] = [];
          let finalResponse: OllamaChatResponse | undefined;
          for await (const chunk of parseNdjsonStream(reader)) {
            if (chunk.message?.content) {
              accumulatedContent += chunk.message.content;
            }
            if (chunk.message?.tool_calls) {
              accumulatedToolCalls.push(...chunk.message.tool_calls);
            }
            if (chunk.done) {
              finalResponse = chunk;
              break;
            }
          }
          if (!finalResponse) {
            throw new Error("Ollama API stream ended without a final response");
          }
          finalResponse.message.content = accumulatedContent;
          if (accumulatedToolCalls.length > 0) {
            finalResponse.message.tool_calls = accumulatedToolCalls;
          }
          const assistantMessage = buildAssistantMessage(finalResponse, {
            api: model.api,
            provider: model.provider,
            id: model.id,
          });
          const reason: Extract<StopReason, "stop" | "length" | "toolUse"> =
            assistantMessage.stopReason === "toolUse" ? "toolUse" : "stop";
          stream.push({ type: "done", reason, message: assistantMessage });
        }
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : String(err);
        stream.push({
          type: "error",
          reason: "error",
          error: {
            role: "assistant" as const,
            content: [],
            stopReason: "error" as StopReason,
            errorMessage,
            api: model.api,
            provider: model.provider,
            model: model.id,
            usage: {
              input: 0,
              output: 0,
              cacheRead: 0,
              cacheWrite: 0,
              totalTokens: 0,
              cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
            },
            timestamp: Date.now(),
          },
        });
      } finally {
        stream.end();
      }
    };
    queueMicrotask(() => void run());
    return stream;
  };
}
