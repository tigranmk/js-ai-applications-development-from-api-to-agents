import * as fs from "node:fs";
import * as path from "node:path";
import * as readline from "node:readline/promises";

import { TextLoader } from "@langchain/classic/document_loaders/fs/text";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

import { OPENAI_API_KEY } from "../commons";

const INDEX_DIR = path.join(__dirname, "microwave_index");
const MANUAL_PATH = path.join(__dirname, "microwave_manual.txt");

const SYSTEM_PROMPT = `You are an expert assistant for the Microwave Oven User Manual (model DW 395 HCG).

Each user message has two parts:
- "##RAG CONTEXT:" contains chunks retrieved from the manual.
- "##USER QUESTION:" contains the actual question from the user.

Rules:
- Answer ONLY from the RAG context and the conversation history.
- If the answer is not covered by them, or the question is unrelated to the microwave oven, politely refuse and say you can only answer questions covered by the microwave manual.
- Do not use outside knowledge or make assumptions beyond the provided context.
- Keep answers concise and accurate.`;

// Template that injects retrieved context and the user question into a single prompt string
const getUserPrompt = (context: string, query: string) => `##RAG CONTEXT:
${context}


##USER QUESTION: 
${query}`;

class MicrowaveRAG {
  vectorStore!: FaissStore;

  /** Resolves when the vector store is fully initialised and ready for queries. */
  readonly ready: Promise<void>;

  constructor(
    public embeddings: OpenAIEmbeddings,
    public llmClient: ChatOpenAI
  ) {
    this.ready = this.setupVectorStore();
  }

  private async setupVectorStore(): Promise<void> {
    if (fs.existsSync(path.join(INDEX_DIR, "faiss.index"))) {
      console.log("📂 Loading existing FAISS index...");
      this.vectorStore = await FaissStore.load(INDEX_DIR, this.embeddings);
    } else {
      console.log("🛠️  Building a new FAISS index...");
      this.vectorStore = await this.createNewIndex();
    }
  }

  private async createNewIndex(): Promise<FaissStore> {
    const documents = await new TextLoader(MANUAL_PATH).load();
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 300,
      chunkOverlap: 40,
      separators: ["\n\n", "\n", ". ", " ", ""],
    });
    const chunks = await splitter.splitDocuments(documents);
    const store = await FaissStore.fromDocuments(chunks, this.embeddings);
    await store.save(INDEX_DIR);
    return store;
  }

  // scoreThreshold is a maximum L2 distance (lower = more similar).
  async retrieveContext(query: string, k = 4, scoreThreshold = 1.0): Promise<string> {
    console.log(`\n🔍 Retrieving top ${k} chunks (max L2 distance ${scoreThreshold})...`);
    const results = await this.vectorStore.similaritySearchWithScore(query, k);

    const relevant = results.filter(([, score]) => score <= scoreThreshold);
    relevant.forEach(([doc, score]) => {
      console.log(`Score (L2 distance): ${score.toFixed(4)}\n${doc.pageContent}\n${"-".repeat(40)}`);
    });
    if (relevant.length === 0) {
      console.log("No chunks passed the score threshold.");
    }

    return relevant.map(([doc]) => doc.pageContent).join("\n\n");
  }

  augmentPrompt(context: string, query: string): string {
    const prompt = getUserPrompt(context, query);
    console.log(`\n🔗 Augmented prompt:\n${prompt}\n`);
    return prompt;
  }

  async generateAnswer(augmentedPrompt: string): Promise<string> {
    const response = await this.llmClient.invoke([
      new SystemMessage(SYSTEM_PROMPT),
      new HumanMessage(augmentedPrompt),
    ]);

    const answer = typeof response.content === "string" ? response.content : JSON.stringify(response.content);
    console.log(`\n🤖 Answer:\n${answer}\n`);
    return answer;
  }
}

const main = async (rag: MicrowaveRAG) => {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  console.log("🎯 Microwave RAG Assistant");
  console.log("Type your question or 'exit' to quit.");

  while (true) {
    const input = await rl.question("➡️  ");

    if (input.trim() === "exit") {
      console.log("Exiting the chat. Goodbye!");
      rl.close();
      process.exit(0);
    }

    if (!input.trim()) continue;

    try {
      const context = await rag.retrieveContext(input);
      const augmentedPrompt = rag.augmentPrompt(context, input);
      await rag.generateAnswer(augmentedPrompt);
    } catch (error) {
      console.error("❌ Error:", error instanceof Error ? error.message : error);
    }
  }
};

const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-small",
  apiKey: OPENAI_API_KEY,
});
const llmClient = new ChatOpenAI({
  model: "gpt-4o",
  temperature: 0,
  apiKey: OPENAI_API_KEY,
});

const rag = new MicrowaveRAG(embeddings, llmClient);
rag.ready.then(() => main(rag)).catch((error) => {
  console.error("❌ Failed to initialise RAG:", error);
  process.exit(1);
});
