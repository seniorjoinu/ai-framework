import { Agent, z } from "@joinu/my-ai-framework";

const AGENT = new Agent({
	apiKey: "",
	baseUrl: new URL("http://localhost"),
	model: "",
});

const response = await AGENT.respond({
	messages: [
		{
			role: "system",
			content: "Are you happy or sad?",
		},
	],
	tools: {
		sad: {
			arg: z.object({ test: z.string() }),
			info: "Call this tool when you're sad",
		},
		happy: {
			arg: z.object({ test: z.string() }),
			info: "Call this tool when you're happy",
		},
	},
});

if (response.happy) {
	console.log("He's happy", response.happy.test);
}

if (response.sad) {
	console.log("He's sad", response.sad.test);
}
