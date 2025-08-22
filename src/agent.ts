import { encodeBase64 } from "jsr:@std/encoding/base64";
import { OpenAI } from "@openai/openai";
import {
	_transformTool,
	tool,
	type ToolDefinition,
	type ToolHandlerInner,
} from "./tools.ts";

/**
 * The chat history.
 */
export type History = OpenAI.ChatCompletionMessageParam[];
/**
 * The response from the model.
 */
export type Response = OpenAI.ChatCompletion;
/**
 * The resolution strategy for tool call errors.
 * - `ignore`: Ignore errors and continue.
 * - `throw`: Throw an error and stop.
 * - `retryAll`: Retry all tool calls up to `maxRetries` times.
 */
export type ResolutionStrategy =
	| { kind: "ignore" }
	| { kind: "throw" }
	| { kind: "retryAll"; maxRetries: number };

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
	/**
	 * The resolution strategy for tool call errors.
	 */
	resolutionStrategy?: ResolutionStrategy;
	/**
	 * The language to use for the model's responses.
	 */
	language?:
		| "English"
		| "Chinese"
		| "Japanese"
		| "Spanish"
		| "German"
		| "Russian";
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
	private defaultInstructions: string[];

	/**
	 * A helper function to define a tool.
	 */
	public static tool: typeof tool = (ctx) => {
		return tool(ctx);
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

		const response = await this.client.chat.completions.create({
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
		});

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
	async respond<T extends Record<string, ToolDefinition>>(args: {
		messages?: History;
		verbosity?: Verbosity;
		tools: T;
	}): Promise<void> {
		const handlers: Record<string, ToolHandlerInner> = {};
		const toolsInner = Object.entries(args.tools).map(([name, t]) => {
			const { tool, handler } = _transformTool(name, t);

			handlers[name] = handler;

			return tool;
		});

		let response = await this._respond(
			toolsInner,
			args.verbosity,
			args.messages
		);
		let verification = await this._verifyAndCallHandlers(response, handlers);

		if (verification === true) {
			return;
		}

		if (
			!this.config.resolutionStrategy ||
			this.config.resolutionStrategy.kind === "ignore"
		) {
			console.warn("Errors during agent response:");
			console.warn(verification);
			console.warn("Ignoring...");

			return;
		}

		if (this.config.resolutionStrategy.kind === "throw") {
			throw new Error(`Errors during agent response:\n\t${verification}`);
		}

		let retryCount = 0;
		while (true) {
			response = await this._respond(toolsInner, args.verbosity, args.messages);
			verification = await this._verifyAndCallHandlers(response, handlers);

			if (verification === true) {
				return;
			}

			retryCount += 1;
			if (retryCount <= this.config.resolutionStrategy.maxRetries) {
				console.warn("Errors during agent response:");
				console.warn(verification);
				console.warn("Retrying...");
				continue;
			}

			throw new Error(
				`[Retry over] Errors during agent response:\n\t${verification}`
			);
		}
	}

	private async _verifyAndCallHandlers(
		response: Response,
		handlers: Record<string, ToolHandlerInner>
	): Promise<string | true> {
		if (response.choices.length === 0) {
			return `The model did not produce anything: ${JSON.stringify(
				response,
				undefined,
				2
			)}`;
		}

		const toolCalls = response.choices[0].message.tool_calls;
		if (!toolCalls) {
			return `The model did not produce any tool calls: ${JSON.stringify(
				response,
				undefined,
				2
			)}`;
		}

		const results: Record<string, string> = {};

		for (const toolCall of toolCalls) {
			if (toolCall.type !== "function") {
				results[
					toolCall.custom.name
				] = `The tool call is not of type "function": ${JSON.stringify(
					toolCall,
					undefined,
					2
				)}`;
				continue;
			}

			const handler = handlers[toolCall.function.name];
			if (!handler) {
				results[
					toolCall.function.name
				] = `The model has called an unknown tool: ${JSON.stringify(
					toolCall,
					undefined,
					2
				)}`;
				continue;
			}

			const errorOrVoid = await handler({
				argJson: toolCall.function.arguments,
				response,
			});

			if (typeof errorOrVoid === "string") {
				results[toolCall.function.name] = errorOrVoid;
				continue;
			}
		}

		const errors = Object.entries(results).map(([k, v]) => `${k}: ${v}`);

		if (errors.length === 0) {
			return true;
		}

		return errors.join("\n\t");
	}

	private async _respond(
		toolsInner: OpenAI.Chat.Completions.ChatCompletionTool[],
		verbosity?: Verbosity,
		messages?: History
	): Promise<Response> {
		const msgs: OpenAI.ChatCompletionMessageParam[] = [
			{
				role: "system",
				content: `
				BASE RULES:
				${this.defaultInstructions.map((it, idx) => `${idx + 1}. ${it}`).join("\n")}
				`,
			},
			...(messages || []),
		];

		const response: Response = await this.client.chat.completions.create({
			model: this.config.model,
			temperature: this.config.temperature,
			top_p: this.config.topP,
			messages: msgs,
			tools: toolsInner,
			verbosity,
		});

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

		this.defaultInstructions = [
			`I only respond with a tool call provided to me by the system. I do NOT respond with plain text. I do NOT call tools, which are not provided to me.`,
			`Before calling a tool, I translate all textual output artifacts into ${
				this.config.language || "English"
			}`,
		];
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
