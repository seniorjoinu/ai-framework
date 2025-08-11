# AI Framework

This is a library that simplifies AI development in Deno. It provides a set of tools for building AI-powered applications, including a type-safe agent that can use tools, and a knowledge base that can be queried and updated.

## Features

- **Type-safe tool definitions:** Define tools with Zod schemas and get full type safety for your tool arguments.
- **Flexible error handling:** Choose how to handle tool call errors with a flexible resolution strategy.
- **B-Tree-like knowledge base:** Store and retrieve information in a knowledge base that works like a B-Tree, but uses an LLM for searching.
- **Automatic document splitting and merging:** The knowledge base automatically splits large documents and merges small ones to optimize performance.

## Installation

This is a Deno project, so there is no installation step. Just import the modules you need from the `src` directory.

## Usage

### Agent

The `Agent` class is the core of the library. It allows you to communicate with the OpenAI API and use tools.

```typescript
import { Agent } from "./src/agent.ts";
import { z } from "zod";

const agent = new Agent({
	apiKey: "YOUR_API_KEY",
	baseUrl: new URL("https://api.openai.com/v1"),
	model: "gpt-4",
});

await agent.respond({
	tools: {
		sayHello: Agent.tool({
			arg: z.object({
				name: z
					.string()
					.meta({ description: "The name of the person to say hello to" }),
			}),
			fn: {
				description: "Says hello to someone.",
				handler: ({ arg }) => {
					console.log(`Hello, ${arg.name}!`);
				},
			},
		}),
	},
});
```

### Knowledge Base

The `Knowledge` class provides a high-level API for a knowledge base that can be queried and updated.

```typescript
import { Knowledge } from "./src/knowledge.ts";

const knowledge = new Knowledge({
	apiKey: "YOUR_API_KEY",
	baseUrl: new URL("https://api.openai.com/v1"),
	model: "gpt-4",
});

// Update the knowledge base
const updateResult = await knowledge.update("The capital of France is Paris.");
console.log(updateResult); // LLM response, something like "Added an information about capital of France"

// Find a document in the knowledge base
const doc = await knowledge.find("What is the capital of France?");
console.log(doc?.content);
```

## Running the Tests

To run the tests, use the following command:

```bash
deno test --allow-env
```
