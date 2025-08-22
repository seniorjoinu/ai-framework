import { assertEquals, assert, assertRejects } from "jsr:@std/assert";
import { z } from "zod";
import { Agent, SerializedAgent } from "./agent.ts";
import { OpenAI } from "@openai/openai";
import { encodeBase64 } from "jsr:@std/encoding/base64";

// Mock OpenAI client
class MockOpenAI {
	chat = {
		completions: {
			create: (args: any): Promise<OpenAI.Chat.Completions.ChatCompletion> => {
				if (
					args.messages &&
					args.messages[0].content[1]?.type === "image_url"
				) {
					return Promise.resolve({
						id: "chatcmpl-123",
						object: "chat.completion",
						created: 1677652288,
						model: "gpt-4-turbo",
						choices: [
							{
								index: 0,
								message: {
									role: "assistant",
									content: "Image description",
									refusal: null,
								},
								logprobs: null,
								finish_reason: "stop",
							},
						],
						usage: {
							prompt_tokens: 20,
							completion_tokens: 20,
							total_tokens: 40,
						},
					});
				}
				return Promise.resolve({
					id: "chatcmpl-123",
					object: "chat.completion",
					created: 1677652288,
					model: "gpt-4-turbo",
					choices: [
						{
							index: 0,
							message: {
								role: "assistant",
								tool_calls: [
									{
										id: "call_123",
										type: "function",
										function: {
											name: "test",
											arguments: JSON.stringify({ test: "world" }),
										},
									},
								],
								refusal: null,
							},
							logprobs: null,
							finish_reason: "tool_calls",
						},
					],
					usage: {
						prompt_tokens: 10,
						completion_tokens: 10,
						total_tokens: 20,
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

Deno.test("Agent.toJSON and Agent.fromJSON should work correctly", () => {
	const agent = new Agent({
		apiKey: "test-key",
		baseUrl: new URL("https://example.com"),
		model: "test-model",
	});

	// @ts-ignore: Accessing private property for testing
	agent.usage = { inputTokens: 100, outputTokens: 200, images: 5 };

	const json = agent.toJSON();
	const newAgent = Agent.fromJSON(json);

	assertEquals(newAgent.toJSON(), json);
});

Deno.test("Agent.parseImage should handle image URL", async () => {
	const mockOpenAI = new MockOpenAI();
	let createArgs: any;
	mockOpenAI.chat.completions.create = (args: any) => {
		createArgs = args;
		return Promise.resolve({
			id: "chatcmpl-123",
			object: "chat.completion",
			created: 1677652288,
			model: "gpt-4-turbo",
			choices: [
				{
					index: 0,
					message: { role: "assistant", content: "A cat", refusal: null },
					logprobs: null,
					finish_reason: "stop",
				},
			],
			usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
		} as any);
	};

	const agent = new Agent(
		{
			apiKey: "test-key",
			baseUrl: new URL("https://example.com"),
			model: "test-model",
		},
		mockOpenAI as any
	);

	const result = await agent.parseImage(
		"http://example.com/cat.jpg",
		"What is this?"
	);
	assertEquals(result, "A cat");
	assertEquals(
		createArgs.messages[0].content[1].image_url.url,
		"http://example.com/cat.jpg"
	);
});

Deno.test("Agent.parseImage should handle raw image data", async () => {
	const mockOpenAI = new MockOpenAI();
	let createArgs: any;
	mockOpenAI.chat.completions.create = (args: any) => {
		createArgs = args;
		return Promise.resolve({
			id: "chatcmpl-123",
			object: "chat.completion",
			created: 1677652288,
			model: "gpt-4-turbo",
			choices: [
				{
					index: 0,
					message: { role: "assistant", content: "A dog", refusal: null },
					logprobs: null,
					finish_reason: "stop",
				},
			],
			usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
		} as any);
	};

	const agent = new Agent(
		{
			apiKey: "test-key",
			baseUrl: new URL("https://example.com"),
			model: "test-model",
		},
		mockOpenAI as any
	);

	const imageBytes = new Uint8Array([1, 2, 3]);
	const result = await agent.parseImage(
		{ bytes: imageBytes, mimeType: "image/png" },
		"What is this?"
	);

	assertEquals(result, "A dog");
	const expectedDataUrl = `data:image/png;base64,${encodeBase64(imageBytes)}`;
	assertEquals(
		createArgs.messages[0].content[1].image_url.url,
		expectedDataUrl
	);
});
