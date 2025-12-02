from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from schemas import AgentResponse

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4")


agent = create_agent(
    #create agent is used instead "create_react_agent".
    model=llm,
    tools=tools,
    response_format=AgentResponse,
    #this will make sure the answer returned will be pydantic object
)

def main():
    result = agent.invoke(
        {
            "messages": [
                {
                "role": "user",
                "content": "Give me 3 AI engineer job openings"
                }
            ]
        }
    )

    # Access structured response from the agent
    structured = result.get("structured_response", None)
    print(structured if structured is not None else result)


if __name__ == "__main__":
    main()
