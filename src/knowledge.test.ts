import { assertEquals, assert } from "jsr:@std/assert";
import {
	Document,
	Heading,
	InMemoryKnowledgeStorage,
	Knowledge,
} from "./knowledge.ts";
import { Agent } from "./agent.ts";
import { z } from "zod";

import { OpenAI } from "@openai/openai";

// Mock OpenAI client
class MockOpenAI {
	chat = {
		completions: {
			create: (_args: any) => {
				return Promise.resolve({} as OpenAI.Chat.Completions.ChatCompletion);
			},
		},
	};
}

// Mock Agent
class MockAgent extends Agent {
	constructor() {
		super(
			{
				apiKey: "test-key",
				baseUrl: new URL("https://example.com"),
				model: "test-model",
			},
			new MockOpenAI() as any
		);
	}

	override respond = (args: any): Promise<void> => {
		if (args.tools.createRoot) {
			const handler = args.tools.createRoot.fn.handler;
			handler({
				arg: { content: "Q: What is the capital of France?\nA: Paris." },
			});
		} else if (args.tools.updateDocument) {
			const handler = args.tools.updateDocument.fn.handler;
			handler({
				arg: {
					newDocument:
						"Q: What is the capital of France?\nA: Paris.\nQ: Where is the Eiffel Tower?\nA: Paris.",
					shortUpdateDescription: "Added a new fact about the Eiffel Tower.",
				},
			});
		} else if (args.tools.selectDocument) {
			const handler = args.tools.selectDocument.fn.handler;
			handler({ arg: 1 });
		}
		return Promise.resolve();
	};
}

Deno.test(
	"InMemoryKnowledgeStorage should store and retrieve documents",
	async () => {
		const storage = new InMemoryKnowledgeStorage();
		const doc: Document = {
			id: 1,
			kind: "d",
			content: "Test document",
			parentId: 0,
		};

		await storage.set(doc);
		const retrieved = await storage.get(1);

		assertEquals(retrieved, doc);
	}
);

Deno.test("Knowledge.update should create a new root document", async () => {
	const storage = new InMemoryKnowledgeStorage();
	const knowledge = new Knowledge(
		{
			apiKey: "test-key",
			baseUrl: new URL("https://example.com"),
			model: "test-model",
			storage,
		},
		new MockAgent()
	);

	const result = await knowledge.update("What is the capital of France?");
	assertEquals(result, "Initialized the knowledge base");

	const root = await storage.get(0);
	assert(root.kind === "d");
	assertEquals(root.content, "Q: What is the capital of France?\nA: Paris.");
	assertEquals(root.parentId, undefined);
});

Deno.test("Knowledge.update should modify an existing document", async () => {
	const storage = new InMemoryKnowledgeStorage();
	await storage.set({
		id: 0,
		kind: "d",
		content: "Q: What is the capital of France?\nA: Paris.",
		parentId: undefined,
	});
	const knowledge = new Knowledge(
		{
			apiKey: "test-key",
			baseUrl: new URL("https://example.com"),
			model: "test-model",
			storage,
		},
		new MockAgent()
	);

	const result = await knowledge.update(
		"Add a new fact: The Eiffel Tower is in Paris."
	);
	assertEquals(result, "Added a new fact about the Eiffel Tower.");

	const root = await storage.get(0);
	assert(root.kind === "d");
	assertEquals(
		root.content,
		"Q: What is the capital of France?\nA: Paris.\nQ: Where is the Eiffel Tower?\nA: Paris."
	);
});

Deno.test(
	"Knowledge.find should traverse headings to find a document",
	async () => {
		const storage = new InMemoryKnowledgeStorage();
		const heading: Heading = {
			id: 0,
			kind: "h",
			refs: [{ id: 1, short: "A test document" }],
			parentId: undefined,
		};
		const doc: Document = {
			id: 1,
			kind: "d",
			content: "The content of the test document",
			parentId: 0,
		};

		await storage.set(heading);
		await storage.set(doc);

		const knowledge = new Knowledge(
			{
				apiKey: "test-key",
				baseUrl: new URL("https://example.com"),
				model: "test-model",
				storage,
			},
			new MockAgent()
		);

		const foundDoc = await knowledge.find("test document");
		assertEquals(foundDoc, doc);
	}
);
