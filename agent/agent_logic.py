import os
from dotenv import load_dotenv
from firecrawl import FirecrawlApp

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# --- THE FIX: We import the modern Gemini-native agent creator ---
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_tavily import TavilySearch
from langchain_experimental.tools import PythonREPLTool
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def scrape_website_with_firecrawl(url: str) -> str:
    cleaned_url = url.strip()
    try:
        app = FirecrawlApp(api_key=os.environ.get("FIRECRAWL_API_KEY"))
        scraped_data = app.scrape_url(cleaned_url)
        if 'markdown' in scraped_data and scraped_data['markdown']:
            return scraped_data['markdown']
        return "Could not extract content."
    except Exception as e: return f"Error during scraping: {e}"

def create_agent_executor():
    """Creates the agent executor using the modern Gemini Tool Calling framework."""
    
    print("🚀 Setting up the Final, Most Reliable Agent System...")
    
    load_dotenv()
    if "GOOGLE_API_KEY" not in os.environ or "TAVILY_API_KEY" not in os.environ or "FIRECRAWL_API_KEY" not in os.environ:
        raise ValueError("Google, Tavily, and Firecrawl API keys must be set.")
    print("✅ API keys loaded.")

    # We use the Pro model for the agent's brain for maximum reliability
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)
    
    print("   -> Initializing RAG tool...")
    persist_directory = 'chroma_db_gemini'
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    # The RAG tool can use the fast 'flash' model
    rag_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
    rag_chain = ({"context": retriever, "input": RunnablePassthrough()} | ChatPromptTemplate.from_template("Answer based on context:\n{context}\nQuestion: {input}") | rag_llm | StrOutputParser())
    
    document_tool = Tool(name="scientific_paper_search", func=rag_chain.invoke, description="Use for questions about the 'Attention Is All You Need' paper.")
    search_tool = TavilySearch(max_results=3)
    python_repl_tool = PythonREPLTool()
    web_reader_tool = Tool(name="web_page_reader", func=scrape_website_with_firecrawl, description="Use to read the full content of a webpage given its URL.")
    
    tools = [document_tool, search_tool, python_repl_tool, web_reader_tool]
    print("✅ All four tools created.")
    
    # --- THE FIX: A new, simpler prompt designed for Tool Calling ---
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a powerful research assistant named Samrat Agent. You must use your tools to find the most accurate and up-to-date information to answer the user's questions. Always be helpful."),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # --- THE FIX: We use the modern 'create_tool_calling_agent' ---
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    print("✅ Modern Gemini Tools Agent created.")
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True
    )
    print("✅ Agent Executor created.")
    
    return agent_executor