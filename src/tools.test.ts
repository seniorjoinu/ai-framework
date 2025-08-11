import { assertEquals, assert } from "jsr:@std/assert";
import { z } from "zod";
import { _transformTool, tool } from "./tools.ts";

Deno.test("tool function should return the same object", () => {
	const toolDef = {
		arg: z.object({ test: z.string() }),
		fn: {
			handler: () => {},
			description: "A test function",
		},
	};
	const result = tool(toolDef);
	assertEquals(result, toolDef);
});

Deno.test(
	"_transformTool should correctly transform a tool definition",
	async () => {
		const toolDef = tool({
			arg: z.object({ test: z.string() }),
			fn: {
				handler: ({ arg }) => {
					assertEquals(arg.test, "hello");
				},
				description: "A test function",
			},
		});

		const { tool: transformedTool, handler } = _transformTool(
			"testTool",
			toolDef
		);

		assert(transformedTool.type === "function");
		assertEquals(transformedTool.function.name, "testTool");
		assertEquals(transformedTool.function.description, "A test function");

		const argJson = JSON.stringify({ test: "hello" });
		const response = {} as any;

		const result = await handler({ argJson, response });
		assertEquals(result, undefined);
	}
);
