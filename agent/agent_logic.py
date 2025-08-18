import os
from dotenv import load_dotenv
# --- THE FIX: We import FirecrawlApp and remove old, unused imports ---
from firecrawl import FirecrawlApp

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_tavily import TavilySearch
from langchain_experimental.tools import PythonREPLTool
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def scrape_website_with_firecrawl(url: str) -> str:
    """Uses Firecrawl to scrape the clean, main content of a webpage given its URL."""
    cleaned_url = url.strip()
    print(f"Scraping URL: {cleaned_url}")
    try:
        app = FirecrawlApp(api_key=os.environ.get("FIRECRAWL_API_KEY"))
        scraped_data = app.scrape_url(cleaned_url)
        
        if 'markdown' in scraped_data and scraped_data['markdown']:
            return scraped_data['markdown']
        return "Could not extract the main content from the page."
    except Exception as e:
        return f"Error during scraping: {e}"

react_prompt_template_str = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do. Your primary goal is to provide a direct, helpful, and friendly answer to the user.
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer. I will present this answer to the user in a natural, conversational way. I will not mention my internal thought process or the tools I used.
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

def create_agent_executor():
    """Creates the complete, final agent executor."""
    print("🚀 Setting up the Final, Polished Agent System...")
    load_dotenv()
    if "GOOGLE_API_KEY" not in os.environ or "TAVILY_API_KEY" not in os.environ or "FIRECRAWL_API_KEY" not in os.environ:
        raise ValueError("Google, Tavily, and Firecrawl API keys must be set.")
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
        func=scrape_website_with_firecrawl,
        description="""Use this to read the full, clean content of a webpage given its URL. This is the best tool for getting detailed, up-to-date information."""
    )
    
    tools = [document_tool, search_tool, python_repl_tool, web_reader_tool]
    print("✅ All four professional-grade tools created.")
    
    prompt = PromptTemplate.from_template(react_prompt_template_str)
    agent = create_react_agent(llm, tools, prompt)
    print("✅ Gemini-powered ReAct agent created with final polished prompt.")
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True
    )
    print("✅ Agent Executor created.")
    
    return agent_executor