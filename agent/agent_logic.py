import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_tavily import TavilySearch
from langchain_experimental.tools import PythonREPLTool
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

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
Thought:{agent_scratchpad}"""

def create_agent_executor():
    print("🚀 Setting up the Final Agent System with Google Gemini...")
    load_dotenv()
    if "GOOGLE_API_KEY" not in os.environ or "TAVILY_API_KEY" not in os.environ:
        raise ValueError("Google and Tavily API keys must be set.")
    print("✅ API keys loaded.")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, convert_system_message_to_human=True)
    print("   -> Initializing RAG tool...")
    persist_directory = 'chroma_db_gemini'
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    rag_chain = ({"context": retriever, "input": RunnablePassthrough()} | PromptTemplate.from_template("Answer based on context:\n\n{context}\n\nQuestion: {input}") | llm | StrOutputParser())
    document_tool = Tool(name="scientific_paper_search", func=rag_chain.invoke, description="""Use for questions about the 'Attention Is All You Need' paper.""")
    search_tool = TavilySearch(max_results=3)
    python_repl_tool = PythonREPLTool()
    tools = [document_tool, search_tool, python_repl_tool]
    print("✅ All three tools created.")
    prompt = PromptTemplate.from_template(react_prompt_template_str)
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    print("✅ Agent Executor created.")
    return agent_executor