import * as readline from "node:readline/promises";

import { Conversation, Message, Role } from "../commons";
import AIClient from "./base_client";


/**
 * Start an interactive chat session with an AI client.
 *
 * This function runs a continuous loop that:
 * 1. Prompts the user for input
 * 2. Sends the conversation history to the AI
 * 3. Displays the AI's response
 * 4. Maintains conversation context
 *
 * The loop continues until the user types 'exit'.
 *
 * @param stream If true, use streaming responses (real-time token display).
 *               If false, use synchronous responses (complete response at once).
 * @param client The AI client instance to use for generating responses.
 */
export async function start(stream: boolean, client: AIClient) {
  const conversation = new Conversation();

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  console.log("💬 Chat started. Type 'exit' to quit.");

  while (true) {
    const input = await rl.question("You: ");

    if (input.trim().toLowerCase() === "exit") {
      console.log("Exiting the chat. Goodbye!");
      break;
    }

    conversation.addMessage(new Message(Role.USER, input));

    const aiMessage = stream
      ? await client.streamResponse(conversation.messages)
      : await client.response(conversation.messages);

    conversation.addMessage(aiMessage);
  }

  rl.close();
}