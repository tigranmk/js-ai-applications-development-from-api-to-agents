import * as fs from "node:fs";
import * as path from "node:path";

import Anthropic from "@anthropic-ai/sdk";

import { ANTHROPIC_API_KEY } from "../../commons";

// https://docs.anthropic.com/en/docs/build-with-claude/vision

const IMAGE_URL = "https://a-z-animals.com/media/2019/11/Elephant-male-1024x535.jpg";
const MODEL = "claude-sonnet-4-5";

const main = async (): Promise<void> => {
  if (!ANTHROPIC_API_KEY) {
    throw new Error("ANTHROPIC_API_KEY cannot be null or empty");
  }

  const client = new Anthropic({ apiKey: ANTHROPIC_API_KEY });

  const logoBase64 = fs.readFileSync(path.join(__dirname, "logo.png")).toString("base64");

  const response = await client.messages.create({
    model: MODEL,
    max_tokens: 1024,
    messages: [
      {
        role: "user",
        content: [
          { type: "image", source: { type: "url", url: IMAGE_URL } },
          { type: "image", source: { type: "base64", media_type: "image/png", data: logoBase64 } },
          { type: "text", text: "Generate poem based on images" },
        ],
      },
    ],
  });

  for (const block of response.content) {
    if (block.type === "text") {
      console.log(block.text);
    }
  }
};

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
