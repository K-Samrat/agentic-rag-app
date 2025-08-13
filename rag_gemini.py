import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def main():
    """
    Main function to set up the RAG pipeline and start an interactive chat session.
    """
    # =======================================================================================
    # SETUP PHASE: This runs only once when the script starts.
    # =======================================================================================
    
    print("🚀 Starting RAG pipeline setup...")

    # --- 1. Load Environment Variables ---
    load_dotenv()
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("GOOGLE_API_KEY not found in .env file. Please create a .env file and add your key.")
    print("✅ Environment variables loaded.")

    # --- 2. Load and Process the Document ---
    script_dir = os.path.dirname(__file__)
    pdf_path = os.path.join(script_dir, "documents", "1706.03762v7.pdf")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file was not found at the specified path: {pdf_path}")
    
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    print(f"✅ Document loaded and split into {len(splits)} chunks.")

    # --- 3. Create Embeddings and Store in Vector Database (ChromaDB) ---
    persist_directory = 'chroma_db_gemini'
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    
    # Check if the vector store is empty, and if so, populate it.
    if not vectorstore.get()['documents']:
        print("Vector store is empty. Populating with new embeddings...")
        vectorstore.add_documents(documents=splits)
        print("✅ Embeddings created and stored successfully.")
    else:
        print("✅ Existing embeddings loaded from ChromaDB.")

    # --- 4. Define the RAG Chain ---
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
    template = """
    You are an expert assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. Be concise.

    Context:
    {context}

    Question:
    {input}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = (
        {"context": retriever, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("✅ RAG chain created.")

    # =======================================================================================
    # INTERACTIVE CHAT PHASE: This loop runs continuously.
    # =======================================================================================

    print("\n\n🤖 RAG system is ready. You can now ask questions about the document.")
    print("   Type 'exit' or 'quit' to end the session.")
    
    while True:
        try:
            # Get user input from the command line
            question = input("\n➡️  Ask a question: ")
            
            # Check if the user wants to exit
            if question.lower() in ["exit", "quit"]:
                print("\n👋 Ending session. Goodbye!")
                break
            
            # If there's a question, invoke the RAG chain
            if question:
                response = rag_chain.invoke(question)
                print("\n💡 Answer:")
                print(response)

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\n\n👋 Ending session. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            break

if __name__ == '__main__':
    main()