import { assertEquals, assert } from "jsr:@std/assert";
import { stub } from "jsr:@std/testing/mock";
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
	const agent = new Agent(
		{
			apiKey: "test-key",
			baseUrl: new URL("https://example.com"),
			model: "test-model",
		},
		new MockOpenAI() as any
	);

	using respondStub = stub(agent, "respond", async (args: any) => {
		if (args.tools.createRoot) {
			await args.tools.createRoot.fn.handler({
				arg: { content: "Q: What is the capital of France?\nA: Paris." },
			});
		}
	});

	const knowledge = new Knowledge(
		{
			apiKey: "test-key",
			baseUrl: new URL("https://example.com"),
			model: "test-model",
			storage,
		},
		agent
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
	const agent = new Agent(
		{
			apiKey: "test-key",
			baseUrl: new URL("https://example.com"),
			model: "test-model",
		},
		new MockOpenAI() as any
	);

	using respondStub = stub(agent, "respond", async (args: any) => {
		if (args.tools.updateDocument) {
			await args.tools.updateDocument.fn.handler({
				arg: {
					newDocument:
						"Q: What is the capital of France?\nA: Paris.\nQ: Where is the Eiffel Tower?\nA: Paris.",
					shortUpdateDescription: "Added a new fact about the Eiffel Tower.",
				},
			});
		} else if (args.tools.selectDocument) {
			await args.tools.selectDocument.fn.handler({ arg: 0 });
		}
	});

	const knowledge = new Knowledge(
		{
			apiKey: "test-key",
			baseUrl: new URL("https://example.com"),
			model: "test-model",
			storage,
		},
		agent
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

		const agent = new Agent(
			{
				apiKey: "test-key",
				baseUrl: new URL("https://example.com"),
				model: "test-model",
			},
			new MockOpenAI() as any
		);

		using respondStub = stub(agent, "respond", async (args: any) => {
			if (args.tools.selectDocument) {
				await args.tools.selectDocument.fn.handler({ arg: 1 });
			}
		});

		const knowledge = new Knowledge(
			{
				apiKey: "test-key",
				baseUrl: new URL("https://example.com"),
				model: "test-model",
				storage,
			},
			agent
		);

		const foundDoc = await knowledge.find("test document");
		assertEquals(foundDoc, doc);
	}
);

Deno.test("Knowledge.update should split a large document", async () => {
	const storage = new InMemoryKnowledgeStorage();
	const largeContent = "a".repeat(200);
	await storage.set({
		id: 0,
		kind: "d",
		content: "Initial content",
		parentId: undefined,
	});

	const agent = new Agent(
		{
			apiKey: "test-key",
			baseUrl: new URL("https://example.com"),
			model: "test-model",
		},
		new MockOpenAI() as any
	);

	let splitCalled = false;
	using respondStub = stub(agent, "respond", async (args: any) => {
		if (args.tools.updateDocument) {
			await args.tools.updateDocument.fn.handler({
				arg: {
					newDocument: largeContent,
					shortUpdateDescription: "Made document large",
				},
			});
		} else if (args.tools.splitDocument) {
			splitCalled = true;
			await args.tools.splitDocument.fn.handler({
				arg: [
					{ content: "a".repeat(100), shortDescription: "Part 1" },
					{ content: "a".repeat(100), shortDescription: "Part 2" },
				],
			});
		} else if (args.tools.selectDocument) {
			await args.tools.selectDocument.fn.handler({ arg: 0 });
		}
	});

	const knowledge = new Knowledge(
		{
			apiKey: "test-key",
			baseUrl: new URL("https://example.com"),
			model: "test-model",
			storage,
			maxDocumentSizeChars: 150,
		},
		agent
	);

	await knowledge.update("make the document large");

	assert(splitCalled, "splitDocument was not called");
	const root = await storage.get(0);
	assert(root.kind === "h", "Root should be a heading after split");
	assertEquals(root.refs.length, 2);

	const child1 = await storage.get(root.refs[0].id);
	assert(child1.kind === "d");
	assertEquals(child1.content, "a".repeat(100));

	const child2 = await storage.get(root.refs[1].id);
	assert(child2.kind === "d");
	assertEquals(child2.content, "a".repeat(100));
});

Deno.test("Knowledge.update should merge small documents", async () => {
	const storage = new InMemoryKnowledgeStorage();
	await storage.set({
		id: 0,
		kind: "h",
		refs: [
			{ id: 1, short: "Doc 1" },
			{ id: 2, short: "Doc 2" },
		],
		parentId: undefined,
	});
	await storage.set({
		id: 1,
		kind: "d",
		content: "Small doc 1",
		parentId: 0,
	});
	await storage.set({
		id: 2,
		kind: "d",
		content: "Small doc 2",
		parentId: 0,
	});

	const agent = new Agent(
		{
			apiKey: "test-key",
			baseUrl: new URL("https://example.com"),
			model: "test-model",
		},
		new MockOpenAI() as any
	);

	let mergeCalled = false;
	using respondStub = stub(agent, "respond", async (args: any) => {
		if (args.tools.selectDocument) {
			await args.tools.selectDocument.fn.handler({ arg: 1 });
		} else if (args.tools.updateDocument) {
			await args.tools.updateDocument.fn.handler({
				arg: {
					newDocument: "updated",
					shortUpdateDescription: "Made document small",
				},
			});
		} else if (args.tools.mergeDocuments) {
			mergeCalled = true;
			await args.tools.mergeDocuments.fn.handler({
				arg: {
					content: "merged content",
					shortDescription: "Merged docs",
				},
			});
		}
	});

	const knowledge = new Knowledge(
		{
			apiKey: "test-key",
			baseUrl: new URL("https://example.com"),
			model: "test-model",
			storage,
			maxDocumentSizeChars: 100, // Set a threshold that will trigger merging
		},
		agent
	);

	await knowledge.update("make a document small");

	assert(mergeCalled, "mergeDocuments was not called");
	const root = await storage.get(0);
	assert(root.kind === "d", "Root should be a document after merge");
	assertEquals(root.content, "merged content");
});

Deno.test(
	"Knowledge.find should return undefined if no document is found",
	async () => {
		const storage = new InMemoryKnowledgeStorage();
		const heading: Heading = {
			id: 0,
			kind: "h",
			refs: [{ id: 1, short: "A test document" }],
			parentId: undefined,
		};
		await storage.set(heading);

		const agent = new Agent(
			{
				apiKey: "test-key",
				baseUrl: new URL("https://example.com"),
				model: "test-model",
			},
			new MockOpenAI() as any
		);

		using respondStub = stub(agent, "respond", async (args: any) => {
			if (args.tools.selectDocument) {
				await args.tools.selectDocument.fn.handler({ arg: -1 });
			}
		});

		const knowledge = new Knowledge(
			{
				apiKey: "test-key",
				baseUrl: new URL("https://example.com"),
				model: "test-model",
				storage,
			},
			agent
		);

		const foundDoc = await knowledge.find(
			"a query for a non-existent document"
		);
		assertEquals(foundDoc, undefined);
	}
);
