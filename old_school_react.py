from dotenv import load_dotenv
from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from typing import Union

from langchain_core.prompts import PromptTemplate
from langchain_core.tools import render_text_description
from langchain_openai import ChatOpenAI

load_dotenv()

from langchain.tools import tool

@tool
def get_text_length(text:str)->int:
    """ Returns the length of the text """
    text= text.strip("'\n").strip(
        "'"
    ) #stripping non alphabetic characters JIC

    return len(text)


if __name__=="__main__":
    print("Hello World")
    tools= [get_text_length]

    template= """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {input}
    Thought: {agent_scratchpad}
        
    """

    prompt= PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        #render_text_description function is formatting nicely, tool names and description
        tool_names= ", ".join([t.name for t in tools]))

    llm= ChatOpenAI(temperature=0, stop="Observation:")
    # the Stop will make LLM stop once it reached the number of token. Otherwise it will guess 1 words after another
    intermediate_steps=[]

    agent= (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: x["agent_scratchpad"],

        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )

    agent_step: Union[AgentAction, AgentFinish]= agent.invoke(
        {
        "input": "What is the length in characters of 'DOG' ?",
        "agent_scratchpad": intermediate_steps,
        }
    )
    print(agent_step)

    if isinstance(agent_step, AgentFinish):
        tool_name= agent_step.tool
        tool_to_use= find_tool_by_name(tools, tool_name)
        tool_input= agent_step.tool_input

        observation = tool_to_use.func(str(tool_input))
        print(f"{observation=}")
        intermediate_steps.append((agent_step, str(observation)))