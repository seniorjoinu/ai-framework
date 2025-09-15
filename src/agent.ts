import { encodeBase64 } from "@std/encoding/base64";
import { OpenAI } from "@openai/openai";
import {
	_transformTool,
	type TooledResponse,
	type ToolDefinition,
} from "./tools.ts";
import type { z } from "@joinu/my-ai-framework";
import { Logger } from "@std/log";

/**
 * The chat history.
 */
export type History = OpenAI.ChatCompletionMessageParam[];
/**
 * The response from the model.
 */
export type Response = OpenAI.ChatCompletion;

/**
 * The configuration for an agent.
 */
export interface AgentConfig {
	/**
	 * The base URL for the OpenAI API.
	 */
	baseUrl: URL;
	/**
	 * The API key for the OpenAI API.
	 */
	apiKey: string;
	/**
	 * The model to use.
	 */
	model: string;
	/**
	 * The temperature to use.
	 */
	temperature?: number;
	/**
	 * The top-p value to use.
	 */
	topP?: number;

	logger?: Logger;
}

/**
 * The serialized state of an agent.
 */
export interface SerializedAgent {
	config: AgentConfig;
	usage: {
		inputTokens: number;
		outputTokens: number;
		images: number;
	};
}

/**
 * The verbosity level for the agent's answers. How verbose the LLM is.
 */
export type Verbosity = "low" | "medium" | "high";

/**
 * An agent that can communicate with the OpenAI API and use tools.
 * This agent ONLY replies with tool calls.
 */
export class Agent {
	private client: OpenAI;
	private usage = {
		inputTokens: 0,
		outputTokens: 0,
		images: 0,
	};

	/**
	 * Creates an agent from a serialized state.
	 * @param obj The serialized agent state.
	 * @returns A new agent instance.
	 */
	public static fromJSON(obj: SerializedAgent): Agent {
		const agent = new Agent(obj.config);
		agent.usage = obj.usage;
		return agent;
	}

	/**
	 * Parses an image and returns the text content from the model's response.
	 * @param image The URL of the image or a raw image object.
	 * @param prompt The prompt to send with the image.
	 * @param detail The detail level for the image.
	 */
	public async parseImage(
		image: string | { bytes: Uint8Array; mimeType: string },
		prompt: string,
		detail?: "low" | "high" | "auto"
	): Promise<string | null> {
		let imageUrl: string;

		if (typeof image === "string") {
			imageUrl = image;
		} else {
			const base64 = encodeBase64(image.bytes);
			imageUrl = `data:${image.mimeType};base64,${base64}`;
		}

		const arg = {
			model: this.config.model,
			messages: [
				{
					role: "user",
					content: [
						{ type: "text", text: prompt },
						{
							type: "image_url",
							image_url: {
								url: imageUrl,
								detail,
							},
						},
					],
				},
			],
		};

		this.config.logger?.debug(
			`[AI] Creating Image Completion for: ${JSON.stringify(arg, undefined, 2)}`
		);

		const response = await this.client.chat.completions.create(arg as any);

		this.config.logger?.debug(
			`[AI] Image Completion response: ${JSON.stringify(
				response,
				undefined,
				2
			)}`
		);

		if (response.usage) {
			this.usage.inputTokens += response.usage.prompt_tokens;
			this.usage.outputTokens += response.usage.completion_tokens;
			this.usage.images += 1;
		}

		return response.choices[0].message.content;
	}

	/**
	 * Responds to a series of messages with tool calls.
	 * @param args The arguments for the response.
	 * @param args.messages The chat history.
	 * @param args.verbosity The verbosity level.
	 * @param args.tools The tools to use.
	 */
	async respond<T extends Record<string, ToolDefinition<z.ZodType>>>(args: {
		messages?: History;
		verbosity?: Verbosity;
		tools: T;
	}): Promise<TooledResponse<T>> {
		const toolsInner = Object.entries(args.tools).map(([name, t]) =>
			_transformTool(name, t)
		);

		const response = await this._respond(
			toolsInner,
			args.verbosity,
			args.messages
		);

		return await this.parseResponse(response, args.tools);
	}

	private async parseResponse<
		T extends Record<string, ToolDefinition<z.ZodType>>
	>(response: Response, tools: T): Promise<TooledResponse<T>> {
		if (response.choices.length === 0) {
			throw new Error(
				`The model did not produce anything: ${JSON.stringify(
					response,
					undefined,
					2
				)}`
			);
		}

		const toolCalls = response.choices[0].message.tool_calls;
		if (!toolCalls) {
			throw new Error(
				`The model did not produce any tool calls: ${JSON.stringify(
					response,
					undefined,
					2
				)}`
			);
		}

		const results: Partial<Record<string, unknown>> = {};

		for (const toolCall of toolCalls) {
			if (toolCall.type !== "function") {
				throw new Error(
					`The tool call is not of type "function": ${JSON.stringify(
						toolCall,
						undefined,
						2
					)}`
				);
			}

			const tool = tools[toolCall.function.name];
			if (!tool) {
				throw new Error(
					`The model has called an unknown tool: ${JSON.stringify(
						toolCall,
						undefined,
						2
					)}`
				);
			}

			const argsJson = JSON.parse(toolCall.function.arguments);
			const args = await tool.arg.parseAsync(argsJson);

			results[toolCall.function.name] = args;
		}

		return results as TooledResponse<T>;
	}

	private async _respond(
		toolsInner: OpenAI.Chat.Completions.ChatCompletionTool[],
		verbosity?: Verbosity,
		messages?: History
	): Promise<Response> {
		const msgs: OpenAI.ChatCompletionMessageParam[] = messages || [];

		const arg = {
			model: this.config.model,
			temperature: this.config.temperature,
			top_p: this.config.topP,
			messages: msgs,
			tools: toolsInner,
			verbosity,
		};

		this.config.logger?.debug(
			`[AI] Creating a completion for: ${JSON.stringify(arg, undefined, 2)}`
		);

		const response: Response = await this.client.chat.completions.create(arg);

		this.config.logger?.debug(
			`[AI] Completion response: ${JSON.stringify(response, undefined, 2)}`
		);

		if (response.usage) {
			this.usage.inputTokens += response.usage.prompt_tokens;
			this.usage.outputTokens += response.usage.completion_tokens;
		}

		return response;
	}

	/**
	 * Creates a new agent.
	 * @param config The configuration for the agent.
	 * @param client The OpenAI client to use.
	 */
	constructor(private config: AgentConfig, client?: OpenAI) {
		this.client =
			client ||
			new OpenAI({
				baseURL: config.baseUrl.toString(),
				apiKey: config.apiKey,
			});
	}

	/**
	 * Serializes the agent's state to a JSON object.
	 * @returns The serialized agent state.
	 */
	public toJSON(): SerializedAgent {
		return {
			config: this.config,
			usage: this.usage,
		};
	}
}
