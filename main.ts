import { Mistral } from "@mistralai/mistralai";
import type { ChatCompletionStreamRequest } from "@mistralai/mistralai/dist/types/chat";
import { ChromaClient, OpenAIEmbeddingFunction } from "chromadb";

import Anthropic from "@anthropic-ai/sdk";

const client = new Mistral({
  apiKey: Deno.env.get("MISTRAL_API_KEY"),
});

const claudeClient = new Anthropic.Anthropic({
  apiKey: Deno.env.get("CLAUDE_API_KEY"),
});

const chromaClient = new ChromaClient();
const collection = await chromaClient.getCollection({
  name: "openai_collections",
  embeddingFunction: new OpenAIEmbeddingFunction({
    openai_api_key: Deno.env.get("OPENAI_API_KEY")!,
    openai_model: "text-embedding-3-small",
  }),
});

async function getSemanticContext(query: string) {
  const embeddings = await client.embeddings.create({
    model: "mistral-embed",
    inputs: [query],
  });

  const queryEmbed = await collection.query({
    nResults: 10,
    queryEmbeddings: embeddings.data[0].embedding!,
  });

  return queryEmbed.documents.flat().join("\n");
}

async function streamResponse(
  messages: ChatCompletionStreamRequest["messages"]
) {
  const stream = await client.chat.stream({
    model: "mistral-large-latest",
    messages,
  });

  for await (const chunk of stream) {
    if (chunk.data.choices[0]?.delta?.content) {
      await Deno.stdout.write(
        new TextEncoder().encode(chunk.data.choices[0].delta.content)
      );
    }
  }
  console.log("\n");
}

async function streamResponseClaude(
  messages: { role: string; content: string }[]
) {
  const messageContent = messages
    .map((msg) => `${msg.role}: ${msg.content}`)
    .join("\n");

  const stream = claudeClient.messages.stream({
    messages: [{ role: "user", content: messageContent }],
    model: "claude-3-opus-20240229",
    max_tokens: 1024,
  });

  for await (const chunk of stream) {
    if (
      chunk.type === "content_block_delta" &&
      chunk.delta.type === "text_delta" &&
      chunk.delta.text
    ) {
      await Deno.stdout.write(new TextEncoder().encode(chunk.delta.text));
    }
  }
  console.log("\n");
}

async function startRepl() {
  const messages = [
    {
      role: "system",
      content:
        "You are a friendly AI assistant who will help user create a JSON schema based on the user's input. If you do not know the answer, you can ask for more information but do not make up information. Make sure you evaluate the user's input and ask for clarification if needed. Also, check the validators schema correct or not. For example, every validator should have a type, a message, and active property. You don't need to write explanations.",
    },
  ];

  while (true) {
    const input = prompt("\nYou: ");
    if (!input) continue;
    if (input.toLowerCase() === "exit") break;

    const context = await getSemanticContext(input);
    messages.push({
      role: "user",
      content: `context: ${context} - Question: ${input}`,
    });

    await streamResponseClaude(messages);
  }
}

// Start the REPL
await startRepl();
