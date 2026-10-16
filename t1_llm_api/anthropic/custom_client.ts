import AIClient from "../base_client";

import { Message, Role } from "../../commons";

/**
 * Custom HTTP client for Anthropic's Claude API.
 *
 * This implementation uses raw HTTP requests instead of the official SDK,
 * demonstrating how to interact with Claude's API directly
 * and handle its Server-Sent Events (SSE) streaming format.
 */
export class CustomAnthropicAIClient extends AIClient {

  /**
   * Get a synchronous response using a raw HTTP POST request.
   *
   * @param messages The conversation history.
   * @returns The AI's response message.
   *
   * Note: Requires 'x-api-key' header and 'anthropic-version' header.
   * Claude's API returns content as an array of content blocks.
   * The response is printed to stdout before being returned.
   */
  response = async (messages: Array<Message>): Promise<Message> => {
    const claudeMessages = messages.map((message) => ({
      role: message.role,
      content: message.content,
    }));

    const res = await fetch(this.endpoint, {
      method: "POST",
      headers: {
        "x-api-key": this.apiKey,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: this.modelName,
        max_tokens: 1024,
        system: this.systemPrompt,
        messages: claudeMessages,
      }),
    });

    if (!res.ok) {
      throw new Error(`Anthropic API error ${res.status}: ${await res.text()}`);
    }

    const data = (await res.json()) as { content: Array<{ type: string; text?: string }> };

    const text = data.content
      .map((block) => (block.type === "text" ? block.text ?? "" : ""))
      .join("");

    console.log(text);

    return new Message(Role.ASSISTANT, text);
  }

  /**
   * Get a streaming response using raw HTTP with Server-Sent Events (SSE).
   *
   * The response is streamed using Anthropic's SSE format, with text deltas
   * printed immediately as they arrive.
   *
   * @param messages The conversation history.
   * @returns The complete AI response message after all deltas are received.
   *
   * Note: Uses Server-Sent Events (SSE) format where each line starts with "data: ".
   * Listens for 'content_block_delta' events with 'text_delta' type.
   * Stops processing when 'message_stop' event is received.
   * Each delta is printed to stdout as it arrives.
   */
  streamResponse = async (messages: Array<Message>): Promise<Message> => {
    const claudeMessages = messages.map((message) => ({
      role: message.role,
      content: message.content,
    }));

    const res = await fetch(this.endpoint, {
      method: "POST",
      headers: {
        "x-api-key": this.apiKey,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: this.modelName,
        max_tokens: 1024,
        system: this.systemPrompt,
        messages: claudeMessages,
        stream: true,
      }),
    });

    if (!res.ok || !res.body) {
      throw new Error(`Anthropic API error ${res.status}: ${await res.text()}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();

    let buffer = "";
    let fullText = "";
    let done = false;

    while (!done) {
      const { done: readerDone, value } = await reader.read();
      done = readerDone;

      buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        if (!line.startsWith("data: ")) {
          continue;
        }

        const event = JSON.parse(line.slice("data: ".length));

        if (event.type === "content_block_delta" && event.delta?.type === "text_delta") {
          process.stdout.write(event.delta.text);
          fullText += event.delta.text;
        }

        if (event.type === "message_stop") {
          done = true;
          break;
        }
      }
    }

    console.log();

    return new Message(Role.ASSISTANT, fullText);
  }
}
