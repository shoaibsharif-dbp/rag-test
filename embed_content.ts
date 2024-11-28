import { OpenAI } from "@openai/openai";
import type { Embedding } from "chromadb";
import { ChromaClient } from "chromadb";
import { Document } from "langchain/document";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

const openai = new OpenAI({
  apiKey: Deno.env.get("OPENAI_API_KEY"),
});

const chromaClient = new ChromaClient();
try {
  await chromaClient.deleteCollection({ name: "openai_collections" });
} catch {
  // ignore
}

const collection = await chromaClient.getOrCreateCollection({
  name: "openai_collections",
});

async function splitDocument(file_name: string) {
  const text = await Deno.readTextFile(file_name);
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 40,
  });

  const document = new Document({ pageContent: text });

  const output = await splitter.splitDocuments([document]);

  return output.map((doc) => doc.pageContent);
}

async function embedContents(contents: string[]) {
  const embeddings = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: contents,
  });
  const embeddingsData = embeddings.data.map((e, index) => ({
    id: index.toString(),
    embedding: e.embedding,
    document: contents[e.index],
  }));

  const ids = embeddingsData.map((e) => e.id?.toString()!);
  const documents = embeddingsData.map((e) => e.document);
  const embeddingsDataMap = embeddingsData.map(
    (e) => e.embedding
  ) as unknown as Embedding[];

  await collection.add({
    embeddings: embeddingsDataMap,
    ids,
    documents,
  });
}

const file_name = "doc.md";

const splittedDocuments = await splitDocument(file_name);
await embedContents(splittedDocuments);
