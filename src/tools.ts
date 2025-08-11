import { z } from "zod";
import { Response } from "./agent.ts";
import OpenAI from "@openai/openai";

/**
 * A tool that can be used by an agent.
 */
export type Tool = OpenAI.ChatCompletionTool;

/**
 * The definition of a tool that can be used by an agent.
 */
export interface ToolDefinition {
	/**
	 * The Zod schema for the arguments of the tool. Use .meta to add a description for the LLM.
	 */
	arg: z.ZodType;
	/**
	 * The function that handles the tool call.
	 */
	fn: {
		/**
		 * The handler function.
		 * @param ctx The context of the tool call.
		 * @param ctx.arg The arguments of the tool call.
		 * @param ctx.response The response from the model.
		 */
		handler: (ctx: { arg: any; response: Response }) => void | Promise<void>;
		/**
		 * A description of the tool.
		 */
		description?: string;
	};
}

/**
 * The inner handler for a tool.
 * @internal
 */
export type ToolHandlerInner = (ctx: {
	argJson: string;
	response: Response;
}) => void | string | Promise<void | string>;

/**
 * A helper function to define a tool.
 * @param obj The tool definition.
 * @returns The tool definition.
 */
export function tool<ReqSchema extends z.ZodType>(obj: {
	/**
	 * The Zod schema for the arguments of the tool.
	 */
	arg: ReqSchema;
	/**
	 * The function that handles the tool call.
	 */
	fn: {
		/**
		 * The handler function.
		 * @param ctx The context of the tool call.
		 * @param ctx.arg The arguments of the tool call.
		 * @param ctx.response The response from the model.
		 */
		handler: (ctx: {
			arg: z.infer<NoInfer<ReqSchema>>;
			response: Response;
		}) => void | Promise<void>;
		/**
		 * A description of the tool.
		 */
		description?: string;
	};
}) {
	return obj as ToolDefinition;
}

/**
 * Transforms a tool definition into a tool and a handler.
 * @param name The name of the tool.
 * @param t The tool definition.
 * @returns The tool and the handler.
 * @internal
 */
export function _transformTool(
	name: string,
	t: ToolDefinition
): { tool: Tool; handler: ToolHandlerInner } {
	const tool: Tool = {
		type: "function",
		function: {
			name: name,
			description: t.fn.description,
			parameters: z.toJSONSchema(t.arg),
			strict: true,
		},
	};

	const handler: ToolHandlerInner = async (ctx) => {
		let arg: any;

		try {
			arg = JSON.parse(ctx.argJson);
		} catch (e) {
			return `The model didn't respond with a valid JSON: ${JSON.stringify(
				e,
				undefined,
				2
			)}`;
		}

		const parseResult = t.arg.safeParse(arg);

		if (!parseResult.success) {
			return `The model has returned invalid tool call args: ${JSON.stringify(
				parseResult.error,
				undefined,
				2
			)}`;
		}

		try {
			const result = t.fn.handler({
				arg: parseResult.data,
				response: ctx.response,
			});

			if (result instanceof Promise) {
				await result;
			}
		} catch (e) {
			return `The tool handler for "${name}" has thrown an eror: ${JSON.stringify(
				e,
				undefined,
				2
			)}`;
		}
	};

	return { tool, handler };
}
