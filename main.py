from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4")

# modern memory API
memory = MemorySaver()

agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=memory
)

def main():
    result = agent.invoke(
        {"messages": [
            {
                "role": "user",
                "content": "Give me 3 AI engineer job openings"}]},
        config={"configurable": {"thread_id": "session-1"}}
    )

    print(result)

if __name__ == "__main__":
    main()
