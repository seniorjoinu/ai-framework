import { assertEquals, assert, assertRejects } from "jsr:@std/assert";
import { z } from "zod";
import { Agent } from "./agent.ts";
import { OpenAI } from "@openai/openai";

// Mock OpenAI client
class MockOpenAI {
	chat = {
		completions: {
			create: (_args: any): Promise<OpenAI.Chat.Completions.ChatCompletion> => {
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

Deno.test(
	"Agent should throw an error when resolutionStrategy is 'throw'",
	async () => {
		const agent = new Agent(
			{
				apiKey: "test-key",
				baseUrl: new URL("https://example.com"),
				model: "test-model",
				resolutionStrategy: { kind: "throw" },
			},
			new MockOpenAI() as any
		);

		await assertRejects(
			() =>
				agent.respond({
					tools: {
						test: Agent.tool({
							arg: z.object({ test: z.string() }),
							fn: {
								description: "A test function",
								handler: () => {
									throw new Error("Tool failed");
								},
							},
						}),
					},
				}),
			Error,
			"Errors during agent response"
		);
	}
);

Deno.test(
	"Agent should ignore errors when resolutionStrategy is 'ignore'",
	async () => {
		const agent = new Agent(
			{
				apiKey: "test-key",
				baseUrl: new URL("https://example.com"),
				model: "test-model",
				resolutionStrategy: { kind: "ignore" },
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
						handler: () => {
							handlerCalled = true;
							throw new Error("Tool failed");
						},
					},
				}),
			},
		});

		assert(handlerCalled, "The tool handler was not called.");
	}
);

Deno.test(
	"Agent should retry on failure when resolutionStrategy is 'retryAll'",
	async () => {
		let attempts = 0;
		const mockOpenAI = new MockOpenAI();
		mockOpenAI.chat.completions.create = (_args: any) => {
			attempts++;
			if (attempts < 2) {
				// Fail first, then succeed
				return Promise.resolve({
					choices: [
						{
							message: {
								tool_calls: [
									{
										type: "function",
										function: {
											name: "test",
											arguments: JSON.stringify({ test: "fail" }),
										},
									},
								],
							},
						},
					],
					usage: { prompt_tokens: 10, completion_tokens: 10 },
				} as OpenAI.Chat.Completions.ChatCompletion);
			}
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
				usage: { prompt_tokens: 10, completion_tokens: 10 },
			} as OpenAI.Chat.Completions.ChatCompletion);
		};

		const agent = new Agent(
			{
				apiKey: "test-key",
				baseUrl: new URL("https://example.com"),
				model: "test-model",
				resolutionStrategy: { kind: "retryAll", maxRetries: 2 },
			},
			mockOpenAI as any
		);

		let handlerCalled = false;
		await agent.respond({
			tools: {
				test: Agent.tool({
					arg: z.object({ test: z.string() }),
					fn: {
						description: "A test function",
						handler: ({ arg }) => {
							if (arg.test === "fail") {
								throw new Error("Tool failed");
							}
							assertEquals(arg.test, "world");
							handlerCalled = true;
						},
					},
				}),
			},
		});

		assertEquals(attempts, 2);
		assert(handlerCalled, "The tool handler was not called on success.");
	}
);

Deno.test("Agent should handle unknown tool calls", async () => {
	const mockOpenAI = new MockOpenAI();
	mockOpenAI.chat.completions.create = (_args: any) => {
		return Promise.resolve({
			choices: [
				{
					message: {
						tool_calls: [
							{
								type: "function",
								function: {
									name: "unknownTool",
									arguments: JSON.stringify({}),
								},
							},
						],
					},
				},
			],
			usage: { prompt_tokens: 10, completion_tokens: 10 },
		} as OpenAI.Chat.Completions.ChatCompletion);
	};

	const agent = new Agent(
		{
			apiKey: "test-key",
			baseUrl: new URL("https://example.com"),
			model: "test-model",
			resolutionStrategy: { kind: "throw" },
		},
		mockOpenAI as any
	);

	await assertRejects(
		() =>
			agent.respond({
				tools: {
					test: Agent.tool({
						arg: z.object({}),
						fn: {
							description: "A test function",
							handler: () => {},
						},
					}),
				},
			}),
		Error,
		"The model has called an unknown tool"
	);
});

Deno.test("Agent should handle multiple tool calls", async () => {
	const mockOpenAI = new MockOpenAI();
	mockOpenAI.chat.completions.create = (_args: any) => {
		return Promise.resolve({
			choices: [
				{
					message: {
						tool_calls: [
							{
								type: "function",
								function: {
									name: "tool1",
									arguments: JSON.stringify({}),
								},
							},
							{
								type: "function",
								function: {
									name: "tool2",
									arguments: JSON.stringify({}),
								},
							},
						],
					},
				},
			],
			usage: { prompt_tokens: 10, completion_tokens: 10 },
		} as OpenAI.Chat.Completions.ChatCompletion);
	};

	const agent = new Agent(
		{
			apiKey: "test-key",
			baseUrl: new URL("https://example.com"),
			model: "test-model",
		},
		mockOpenAI as any
	);

	let tool1Called = false;
	let tool2Called = false;
	await agent.respond({
		tools: {
			tool1: Agent.tool({
				arg: z.object({}),
				fn: {
					description: "Tool 1",
					handler: () => {
						tool1Called = true;
					},
				},
			}),
			tool2: Agent.tool({
				arg: z.object({}),
				fn: {
					description: "Tool 2",
					handler: () => {
						tool2Called = true;
					},
				},
			}),
		},
	});

	assert(tool1Called, "Tool 1 was not called.");
	assert(tool2Called, "Tool 2 was not called.");
});

Deno.test("Agent should handle no tool calls", async () => {
	const mockOpenAI = new MockOpenAI();
	mockOpenAI.chat.completions.create = (_args: any) => {
		return Promise.resolve({
			choices: [
				{
					message: {
						tool_calls: null,
					},
				},
			],
			usage: { prompt_tokens: 10, completion_tokens: 10 },
		} as unknown as OpenAI.Chat.Completions.ChatCompletion);
	};

	const agent = new Agent(
		{
			apiKey: "test-key",
			baseUrl: new URL("https://example.com"),
			model: "test-model",
			resolutionStrategy: { kind: "throw" },
		},
		mockOpenAI as any
	);

	await assertRejects(
		() =>
			agent.respond({
				tools: {
					test: Agent.tool({
						arg: z.object({}),
						fn: {
							description: "A test function",
							handler: () => {},
						},
					}),
				},
			}),
		Error,
		"The model did not produce any tool calls"
	);
});

Deno.test("Agent should pass configuration to OpenAI client", async () => {
	const mockOpenAI = new MockOpenAI();
	let createArgs: any;
	mockOpenAI.chat.completions.create = (args: any) => {
		createArgs = args;
		return Promise.resolve({
			choices: [
				{
					message: {
						tool_calls: [],
					},
				},
			],
			usage: { prompt_tokens: 10, completion_tokens: 10 },
		} as unknown as OpenAI.Chat.Completions.ChatCompletion);
	};

	const agent = new Agent(
		{
			apiKey: "test-key",
			baseUrl: new URL("https://example.com"),
			model: "test-model",
			temperature: 0.5,
			topP: 0.5,
			language: "German",
		},
		mockOpenAI as any
	);

	await agent.respond({
		tools: {},
	});

	assertEquals(createArgs.temperature, 0.5);
	assertEquals(createArgs.top_p, 0.5);
	assert(createArgs.messages[0].content.includes("German"));
});

Deno.test("Agent should track token usage", async () => {
	const agent = new Agent(
		{
			apiKey: "test-key",
			baseUrl: new URL("https://example.com"),
			model: "test-model",
		},
		new MockOpenAI() as any
	);

	await agent.respond({
		tools: {
			test: Agent.tool({
				arg: z.object({}),
				fn: {
					description: "A test function",
					handler: () => {},
				},
			}),
		},
	});

	// @ts-ignore: Accessing private property for testing
	const usage = agent.usage;
	assertEquals(usage.inputTokens, 10);
	assertEquals(usage.outputTokens, 10);
});
