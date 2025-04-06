from datetime import datetime

from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain.tools import Tool
from langchain_community.utilities import WikipediaAPIWrapper

# 1. search the web tool
search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search_the_web",
    description="Search the web for information.",
    func=search.run,
)

# 2. Wikipedia tool

wikipedia_api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=wikipedia_api_wrapper)

# 3. custom report tool

def save_to_txt(data: str, filename: str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)

    return f"Data successfully saved to {filename}"

save_tool = Tool.from_function(
    func=save_to_txt,
    name="save_text_to_file",
    description="save structured research data to a text file. Input should be the data to save.",
)