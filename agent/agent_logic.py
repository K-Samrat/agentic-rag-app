import os
from dotenv import load_dotenv

# --- THE FIX: We are enabling LangChain's debug mode ---
import langchain
langchain.debug = True

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_tavily import TavilySearch
from langchain_experimental.tools import PythonREPLTool
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def create_agent_executor():
    """Creates the complete, final agent executor with a persona."""
    
    print("🚀 Setting up the Final Persona Agent System...")
    
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
    rag_chain = ({"context": retriever, "input": RunnablePassthrough()} | ChatPromptTemplate.from_template("Answer based on context:\n{context}\nQuestion: {input}") | llm | StrOutputParser())
    
    document_tool = Tool(
        name="scientific_paper_search",
        func=rag_chain.invoke,
        description="""Use this tool for specific questions about the 'Attention Is All You Need' research paper."""
    )
    search_tool = TavilySearch(max_results=3)
    python_repl_tool = PythonREPLTool()
    tools = [document_tool, search_tool, python_repl_tool]
    print("✅ All three tools created.")
    
    persona_prompt_str = """
    You are Samrat Agent, a world-class research assistant. Your goal is to provide accurate, well-structured, and helpful answers to the user.
    Here are your instructions:
    1.  **Always be helpful and proactive:** If you can find the answer, you must provide it. Never claim you don't know something if you have a tool that can find the answer.
    2.  **Use your tools:** You have access to a set of powerful tools. You must decide which tool is best for the user's question and use it.
    3.  **Structure your answers:** Break down complex answers into sections. Use **bold headings** and **numbered or bulleted lists** to make your answers easy to read.
    4.  **Synthesize information:** Do not just copy-paste tool outputs. You must read the information from your tools and then synthesize a final, comprehensive answer in your own words.
    5.  **Be conversational:** Maintain a friendly and professional tone.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", persona_prompt_str),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    print("✅ Persona-driven agent created.")
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True
    )
    print("✅ Agent Executor created.")
    
    return agent_executor