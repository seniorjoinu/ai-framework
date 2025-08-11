import { z } from "zod";
import { Response } from "./agent.ts";
import OpenAI from "@openai/openai";

export type Tool = OpenAI.ChatCompletionTool;

export interface ToolDefinition {
	arg: z.ZodType;
	fn: {
		handler: (ctx: { arg: any; response: Response }) => void | Promise<void>;
		description?: string;
	};
}

export type ToolHandlerInner = (ctx: {
	argJson: string;
	response: Response;
}) => void | string | Promise<void | string>;

export function tool<ReqSchema extends z.ZodType>(obj: {
	arg: ReqSchema;
	fn: {
		handler: (ctx: {
			arg: z.infer<NoInfer<ReqSchema>>;
			response: Response;
		}) => void | Promise<void>;
		description?: string;
	};
}) {
	return obj as ToolDefinition;
}

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
			const result = t.fn.handler(arg);

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
