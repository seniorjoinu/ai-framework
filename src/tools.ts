import { z } from "zod";
import type OpenAI from "@openai/openai";

/**
 * A tool that can be used by an agent.
 */
export type Tool = OpenAI.ChatCompletionTool;

/**
 * The definition of a tool that can be used by an agent.
 */
export interface ToolDefinition<A extends z.ZodType> {
	/**
	 * The Zod schema for the arguments of the tool. Use .meta to add a description for the LLM.
	 */
	arg: A;
	/**
	 * Description of the tool to call.
	 */
	info: string;
}

/**
 * Transforms a tool definition into a tool and a handler.
 * @param name The name of the tool.
 * @param t The tool definition.
 * @returns The tool and the handler.
 * @internal
 */
export function _transformTool<
	A extends z.ZodType,
	T extends ToolDefinition<A>
>(name: string, t: T): Tool {
	return {
		type: "function",
		function: {
			name: name,
			description: t.info,
			parameters: z.toJSONSchema(t.arg),
			strict: true,
		},
	};
}

export type TooledResponse<
	T extends Record<string, ToolDefinition<z.ZodType>>
> = Partial<{
	[K in keyof T]: z.infer<T[K]["arg"]>;
}>;
