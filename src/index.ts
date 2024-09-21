import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { AIMessage } from "@langchain/core/messages";
import {
  END,
  MessagesAnnotation,
  START,
  StateGraph,
} from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { ChatOpenAI } from "@langchain/openai";

const systemMessage = {
  role: 'system',
  content:
    `You are a search assistant. Your main job is to look for relevant information on the internet.
    When asked a question, use the available search tools to find accurate and up-to-date information.
    Provide concise and informative responses based on the search results.
    If you need more information to answer a question completely, don't hesitate to perform additional searches.`,
};

const llm = new ChatOpenAI({
  model: 'gpt-4o-mini',
  temperature: 0,
});

const webSearchTool = new TavilySearchResults({
  maxResults: 3,
});
const tools = [webSearchTool];

const toolNode = new ToolNode(tools);

const callModel = async (state: typeof MessagesAnnotation.State) => {
  const { messages } = state;

  const llmWithTools = llm.bindTools(tools);
  const result = await llmWithTools.invoke([systemMessage, ...messages]);
  return { messages: [result] };
};

const shouldContinue = (state: typeof MessagesAnnotation.State) => {
  const { messages } = state;

  const lastMessage = messages[messages.length - 1];
  if (
    lastMessage._getType() !== "ai" ||
    !(lastMessage as AIMessage).tool_calls?.length
  ) {

    return END;
  }
  return "tools";
};

const workflow = new StateGraph(MessagesAnnotation)
  .addNode("agent", callModel)
  .addEdge(START, "agent")
  .addNode("tools", toolNode)
  .addEdge("tools", "agent")
  .addConditionalEdges("agent", shouldContinue, ["tools", END]);

export const graph = workflow.compile({
  // The LangGraph Studio/Cloud API will automatically add a checkpointer
  // only uncomment if running locally
  // checkpointer: new MemorySaver(),
});
