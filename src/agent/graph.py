import random 
from typing import Literal
from typing_extensions import TypedDict

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode



# State is imported from langgraph.graph


# Defining the LLM
llm = ChatOpenAI(model="gpt-4o")

node1_msg = SystemMessage(content="You are a pilot. Analyze what the user has said so far")
node2_msg = SystemMessage(content="You are a doctor. Analyze what the user has said so far")
node3_msg = SystemMessage(content="You are a sociology major. Analyze what the user has said so far")



# Conditional edge
def decide_mood(MessagesState) -> Literal["node_2", "node_3"]:
    
    # Often, we will use state to decide on the next node to visit
    user_input = MessagesState['graph_state'] 
    
    # Here, let's just do a 50 / 50 split between nodes 2, 3
    if random.random() < 0.5:

        # 50% of the time, we return Node 2
        return "node_2"
    
    # 50% of the time, we return Node 3
    return "node_3"

# Nodes
def node_1(MessagesState):
    print("---Node 1---")
    #return {"graph_state":MessagesState['graph_state'] +" I am"}
    return {"messages": [llm.invoke([node1_msg] + state["messages"])]}

def node_2(MessagesState):
    print("---Node 2---")
    #return {"graph_state":MessagesState['graph_state'] +" happy!"}
    return {"messages": [llm.invoke([node2_msg] + state["messages"])]}

def node_3(MessagesState):
    print("---Node 3---")
    #return {"graph_state":MessagesState['graph_state'] +" sad"}
    return {"messages": [llm.invoke([node3_msg] + state["messages"])]}

def node_4(MessagesState): 
    print("---Node 4---")
    return {"graph_state":MessagesState['graph_state'] +", this program is ending."}



# Build graph
builder = StateGraph(MessagesState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)
builder.add_node("node_4", node_4)

builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", "node_4")
builder.add_edge("node_4",END)

# Compile graph
graph = builder.compile()
