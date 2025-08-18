import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_tavily import TavilySearch
from langchain_experimental.tools import PythonREPLTool
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def scrape_website_content(url: str) -> str:
    """Scrapes the main text content from a given URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        main_content = soup.find('main') or soup.find('article') or soup.body
        if main_content:
            return ' '.join(main_content.get_text().split())
        return "Could not find main content."
    except requests.RequestException as e:
        return f"Error fetching URL: {e}"

# --- THIS IS THE FINAL PROMPT ---
react_prompt_template_str = """
Answer the following questions as best you can. You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do.
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

def create_agent_executor():
    """Creates the complete, final agent executor."""
    print("🚀 Setting up the Final, Resilient Agent System...")
    load_dotenv()
    if "GOOGLE_API_KEY" not in os.environ or "TAVILY_API_KEY" not in os.environ:
        raise ValueError("Google and Tavily API keys must be set.")
    print("✅ API keys loaded.")

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
    
    print("   -> Initializing RAG tool...")
    persist_directory = 'chroma_db_gemini'
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    rag_chain = ({"context": retriever, "input": RunnablePassthrough()} | PromptTemplate.from_template("Answer based on context:\n{context}\nQuestion: {input}") | llm | StrOutputParser())
    
    document_tool = Tool(
        name="scientific_paper_search",
        func=rag_chain.invoke,
        description="""Use this tool for specific questions about the 'Attention Is All You Need' research paper ONLY."""
    )
    search_tool = TavilySearch(max_results=3)
    python_repl_tool = PythonREPLTool()
    web_reader_tool = Tool(
        name="web_page_reader",
        func=scrape_website_content,
        description="""Use this to read the full text content of a webpage given its URL. This is useful after a 'tavily_search' to get more details."""
    )
    
    tools = [document_tool, search_tool, python_repl_tool, web_reader_tool]
    print("✅ All four tools created.")
    
    # --- Agent Creation with the new, resilient prompt ---
    # We add the "Plan B" instruction to the main prompt template.
    resilient_prompt_str = react_prompt_template_str.replace(
        "Thought: you should always think about what to do.",
        """Thought: you should always think about what to do.
For questions requiring up-to-date information, your primary plan is a two-step process:
1. Use the 'tavily_search' tool to find relevant URLs.
2. Use the 'web_page_reader' tool on the most promising URL.
However, if the 'web_page_reader' tool fails for any reason, you MUST NOT give up. You must fall back to your Plan B, which is to formulate a final answer using only the information from the 'tavily_search' results."""
    )
    prompt = PromptTemplate.from_template(resilient_prompt_str)
    
    agent = create_react_agent(llm, tools, prompt)
    print("✅ Gemini-powered ReAct agent created with resilient prompt.")
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True
    )
    print("✅ Agent Executor created.")
    
    return agent_executor