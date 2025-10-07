from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode

def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtracts b from a.

    Args:
        a: first int
        b: second int
    """
    return a - b


def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

def simple_interest(principal: int, rate: int, time: int) -> float:
    """ Multiply principal, rate and time using the formula P x rate x time 
    Args:
        principal: Amount to be repaid 
        rate: Rate charged by the financial institution per annum or year 
        time: Time to repay the loan
    
    """
    return principal * rate * time / 100



def compound_interest(principal: int, rate: int, time: int, n: int) -> float:
    """
    Principal: The principal amount (the initial amount of money).
    rate: The annual interest rate (provided to the function as an integer).
    n: The number of times that interest is compounded per year (provided to the function as an integer. If not provided, assume 1).
    time: The number of years the money is invested or borrowed for (provided to the function as an integer).

    The formula used to calculate is Compound interest rate = (1 + r/n)^nt 

    """
    rate = rate/100
    return principal * (1 + rate/n)**(n*time)



tools = [add, subtract, multiply, divide, simple_interest, compound_interest]

# Define LLM with bound tools
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)

# System message
sys_msg = SystemMessage(content="""
You are a financial expert tasked with helping users determine the interest rate for their loans. You understand which type of loan requires compound interest calculation and which type requires simple interest. 

In the input provided, do we need to calculate simple interest or compound interest calculation? If so, why? 

If a rate of interest is provided, run a validation to check if it is in the right % format. Rates are always in the range 0-1 and are decimals in nature.
""")

#You are a helpful assistant tasked with writing performing arithmetic on a set of inputs. 
#READ THE QUESTIONS CAREFULLY
#If it is arithmetic, automatically use the rules of BODMAS. 


# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")

# Compile graph
graph = builder.compile()
