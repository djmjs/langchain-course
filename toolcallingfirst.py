from typing import List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.tools import tool, BaseTool
from langchain_openai import ChatOpenAI

from callbacks import AgentCallbackHandler

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip(
        '"'
    )  # stripping away non alphabetic characters just in case

    return len(text)


def find_tool_by_name(tools: List[BaseTool], tool_name: str) -> BaseTool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool wtih name {tool_name} not found")


if __name__ == "__main__":
    print("Hello LangChain Tools (.bind_tools)!")
    tools = [get_text_length]

    llm = ChatOpenAI(
        temperature=0,
        callbacks=[AgentCallbackHandler()],
        #it allows you to monitor, log, or respond to events during the model’s operation
        # created another py file for it
    )
    llm_with_tools = llm.bind_tools(tools)
    #bind_tools is the method to equip your llm with the tools. It takes tool des and arg.
    # every req will be appended and llm will make a desicion to make a call or not

    # Start conversation
    messages = [HumanMessage(content="What is the length of the word: DOG")]

    while True:
        ai_message = llm_with_tools.invoke(messages)

        # If the model decides to call tools, execute them and return results
        tool_calls = getattr(ai_message, "tool_calls", None) or []
        # this supports multiple (parralel) tool calls. It gives a list of tools or none based on need
        if len(tool_calls) > 0:
            # if one above gives 1+ tools then runs what is below
            messages.append(ai_message)
            for tool_call in tool_calls:
                # tool_call is typically a dict with keys: id, type, name, args
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_call_id = tool_call.get("id")

                tool_to_use = find_tool_by_name(tools, tool_name)
                observation = tool_to_use.invoke(tool_args)
                print(f"observation={observation}")

                messages.append(
                    ToolMessage(content=str(observation), tool_call_id=tool_call_id)
                )
            # Continue loop to allow the model to use the observations
            continue

        # No tool calls -> final answer
        print(ai_message.content)
        break