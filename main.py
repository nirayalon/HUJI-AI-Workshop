import os
from typing import List

from dotenv import load_dotenv
from langchain.agents import Agent
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from pydantic import BaseModel

from tools import search_tool, wiki_tool, save_tool

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: List[str]
    tool_used: List[str]


parser = PydanticOutputParser(pydantic_object=ResearchResponse)
format_instructions = parser.get_format_instructions()


prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
        """
        You are a research assistant that will help generate a research paper answer the user query and use the necessary tools. 
        Wrap the output in the following format and provide no other text:\n{format_instructions}
        provide a link to the source of the information in the sources field.
        """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=format_instructions)


llm = ChatOpenAI(
    model="gpt-4-turbo",
    api_key=os.environ.get("OPENAI_API_KEY"),
    temperature=0,
)

tools = [search_tool, wiki_tool]

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)

user_query = input("What can I help you to research? ")
raw_response = agent_executor.invoke({"query": user_query})

try:
    structured_response = parser.parse(raw_response.get("output"))
    print("Structured Output\n")
    print(f"Topic={structured_response.topic}")
    print(f"Summary={structured_response.summary}")
    print(f"Sources={structured_response.sources}")
    print(f"Tools={structured_response.tool_used}")
except Exception as e:
    print("Error parsing the output:", e)
    print("Raw output:", raw_response.get("output"))

