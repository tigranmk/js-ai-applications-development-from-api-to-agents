import Anthropic from "@anthropic-ai/sdk";

import AIClient from "../base_client";

import { Message, Role } from "../../commons";

/**
 * Client for Anthropic's Claude API using the official SDK.
 *
 * This implementation uses the official Anthropic TypeScript library to interact
 * with Claude models, providing both synchronous and streaming response capabilities.
 *
 * Inherits all attributes from AIClient.
 */
export class AnthropicAIClient extends AIClient {
  client!: Anthropic;

  /**
   * Initialize the Anthropic client with the official SDK.
   *
   * @param args Constructor parameters inherited from AIClient (endpoint, modelName, apiKey, systemPrompt).
   */
  constructor(...args: ConstructorParameters<typeof AIClient>) {
    super(...args);
    this.client = new Anthropic({
      apiKey: this.apiKey,
    });
  }

  /**
   * Get a synchronous response from Anthropic's Claude API.
   *
   * @param messages The conversation history.
   * @returns The AI's response message.
   *
   * Note: Claude's API uses a separate 'system' parameter for system instructions.
   * Response content blocks are concatenated into a single text response.
   * The response is printed to stdout before being returned.
   */
  response = async (messages: Array<Message>): Promise<Message> => {
    const claudeMessages = messages.map((message) => ({
      role: message.role as "user" | "assistant",
      content: message.content,
    }));

    const response = await this.client.messages.create({
      model: this.modelName,
      max_tokens: 1024,
      system: this.systemPrompt,
      messages: claudeMessages,
    });

    const text = response.content
      .map((block) => (block.type === "text" ? block.text : ""))
      .join("");

    console.log(text);

    return new Message(Role.ASSISTANT, text);
  };

  /**
   * Get a streaming response from Anthropic's Claude API.
   *
   * The response is streamed using event-based streaming, with text deltas
   * printed immediately as they arrive.
   *
   * @param messages The conversation history.
   * @returns The complete AI response message after all deltas are received.
   *
   * Note: Listens for 'text' events with text deltas.
   * Each delta is printed to stdout as it arrives for real-time display.
   */
  streamResponse = async (messages: Array<Message>): Promise<Message> => {
    const claudeMessages = messages.map((message) => ({
      role: message.role as "user" | "assistant",
      content: message.content,
    }));

    const stream = this.client.messages.stream({
      model: this.modelName,
      max_tokens: 1024,
      system: this.systemPrompt,
      messages: claudeMessages,
    });

    stream.on("text", (textDelta) => {
      process.stdout.write(textDelta);
    });

    const finalMessage = await stream.finalMessage();
    console.log();

    const text = finalMessage.content
      .map((block) => (block.type === "text" ? block.text : ""))
      .join("");

    return new Message(Role.ASSISTANT, text);
  };
}
