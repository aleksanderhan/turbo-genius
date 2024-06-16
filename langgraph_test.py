import json

from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, MessageGraph
from langgraph.prebuilt.tool_node import ToolNode


# Define the function that determines whether to continue or not
def should_continue(messages):
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return END
    else:
        return "action"


# Define a new graph
workflow = MessageGraph()

tools = [TavilySearchResults(max_results=1)]
model = ChatHuggingFace(model="claude-3-haiku-20240307").bind_tools(tools)
workflow.add_node("agent", model)
workflow.add_node("action", ToolNode(tools))

workflow.set_entry_point("agent")

# Conditional agent -> action OR agent -> END
workflow.add_conditional_edges(
    "agent",
    should_continue,
)

# Always transition `action` -> `agent`
workflow.add_edge("action", "agent")

memory = SqliteSaver.from_conn_string(":memory:") # Here we only save in-memory

# Setting the interrupt means that any time an action is called, the machine will stop
app = workflow.compile(checkpointer=memory, interrupt_before=["action"])