import { assertEquals, assert } from "jsr:@std/assert";
import { z } from "zod";
import { Agent } from "./agent.ts";
import { OpenAI } from "@openai/openai";

// Mock OpenAI client
class MockOpenAI {
	chat = {
		completions: {
			create: (_args: any) => {
				return Promise.resolve({
					choices: [
						{
							message: {
								tool_calls: [
									{
										type: "function",
										function: {
											name: "test",
											arguments: JSON.stringify({ test: "world" }),
										},
									},
								],
							},
						},
					],
					usage: {
						prompt_tokens: 10,
						completion_tokens: 10,
					},
				} as OpenAI.Chat.Completions.ChatCompletion);
			},
		},
	};
}

Deno.test("Agent should call the correct tool handler", async () => {
	const agent = new Agent(
		{
			apiKey: "test-key",
			baseUrl: new URL("https://example.com"),
			model: "test-model",
		},
		new MockOpenAI() as any
	);

	let handlerCalled = false;
	await agent.respond({
		tools: {
			test: Agent.tool({
				arg: z.object({ test: z.string() }),
				fn: {
					description: "A test function",
					handler: ({ arg }) => {
						assertEquals(arg.test, "world");
						handlerCalled = true;
					},
				},
			}),
		},
	});

	assert(handlerCalled, "The tool handler was not called.");
});
