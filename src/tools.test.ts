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

Deno.test("_transformTool should handle invalid JSON", async () => {
	const toolDef = tool({
		arg: z.object({ test: z.string() }),
		fn: {
			handler: () => {},
			description: "A test function",
		},
	});

	const { handler } = _transformTool("testTool", toolDef);

	const argJson = "{ test: 'hello' }";
	const response = {} as any;

	const result = await handler({ argJson, response });
	assert(typeof result === "string");
	assert(result.includes("The model didn't respond with a valid JSON"));
});

Deno.test("_transformTool should handle schema validation errors", async () => {
	const toolDef = tool({
		arg: z.object({ test: z.string() }),
		fn: {
			handler: () => {},
			description: "A test function",
		},
	});

	const { handler } = _transformTool("testTool", toolDef);

	const argJson = JSON.stringify({ test: 123 });
	const response = {} as any;

	const result = await handler({ argJson, response });
	assert(typeof result === "string");
	assert(result.includes("The model has returned invalid tool call args"));
});

Deno.test("_transformTool should handle handler exceptions", async () => {
	const toolDef = tool({
		arg: z.object({ test: z.string() }),
		fn: {
			handler: () => {
				throw new Error("Handler error");
			},
			description: "A test function",
		},
	});

	const { handler } = _transformTool("testTool", toolDef);

	const argJson = JSON.stringify({ test: "hello" });
	const response = {} as any;

	const result = await handler({ argJson, response });
	assert(typeof result === "string");
	assert(result.includes('The tool handler for "testTool" has thrown an eror'));
});
