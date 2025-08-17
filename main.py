import json
from fastapi import FastAPI, Depends, HTTPException, Header
from typing import Annotated
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

# Import the specific error for graceful handling
from google.api_core.exceptions import ResourceExhausted

from agent.agent_logic import create_agent_executor
from langchain_google_genai import ChatGoogleGenerativeAI
import database
import models

app = FastAPI(title="Persistent Agentic RAG System API", version="5.0.0")
app.state.agent_executor = None
app.state.title_generator_llm = None
async def get_db():
    async with database.AsyncSessionLocal() as session:
        yield session
@app.on_event("startup")
async def startup_event():
    print("...Loading agent executor and setting up database...")
    app.state.agent_executor = create_agent_executor()
    app.state.title_generator_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
    async with database.engine.begin() as conn:
        await conn.run_sync(models.Base.metadata.create_all)
    print("...Startup complete.")

class Query(BaseModel): question: str
class MessageResponse(BaseModel): id: int; role: str; content: str
class ConversationListItem(BaseModel): id: int; title: str
class DeleteStatus(BaseModel): status: str
class ConversationResponse(BaseModel): id: int; title: str; messages: list[MessageResponse]

@app.post("/conversations/", response_model=ConversationListItem)
async def create_conversation(x_session_id: Annotated[str, Header()], db: AsyncSession = Depends(get_db)):
    new_convo = models.Conversation(session_id=x_session_id)
    db.add(new_convo); await db.commit(); await db.refresh(new_convo)
    return new_convo
@app.get("/conversations/", response_model=list[ConversationListItem])
async def get_all_conversations(x_session_id: Annotated[str, Header()], db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(models.Conversation).where(models.Conversation.session_id == x_session_id).order_by(models.Conversation.id.desc()))
    return result.scalars().all()
@app.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: int, x_session_id: Annotated[str, Header()], db: AsyncSession = Depends(get_db)):
    query = (select(models.Conversation).options(selectinload(models.Conversation.messages)).where(models.Conversation.id == conversation_id, models.Conversation.session_id == x_session_id))
    result = await db.execute(query); convo = result.scalar_one_or_none()
    if convo is None: raise HTTPException(status_code=404, detail="Conversation not found or not owned by session")
    return convo
@app.delete("/conversations/{conversation_id}", response_model=DeleteStatus)
async def delete_conversation(conversation_id: int, x_session_id: Annotated[str, Header()], db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(models.Conversation).where(models.Conversation.id == conversation_id, models.Conversation.session_id == x_session_id))
    convo = result.scalar_one_or_none()
    if convo:
        await db.delete(convo); await db.commit(); return {"status": "success"}
    raise HTTPException(status_code=404, detail="Conversation not found or not owned by session")

async def stream_and_save(conversation_id: int, question: str, session_id: str, db: AsyncSession):
    result = await db.execute(select(models.Conversation).where(models.Conversation.id == conversation_id, models.Conversation.session_id == session_id))
    convo = result.scalar_one_or_none()
    if not convo: yield "Error: Conversation not found or access denied."; return
    user_message = models.Message(role="user", content=question, conversation_id=conversation_id)
    db.add(user_message); await db.commit()
    
    try:
        query = (select(models.Conversation).options(selectinload(models.Conversation.messages)).where(models.Conversation.id == conversation_id))
        result = await db.execute(query)
        convo = result.scalar_one()
        if convo.title == "New Chat" and len(convo.messages) == 1:
            title_prompt = f"Summarize the following user query into a short, concise title of 5 words or less: '{question}'"
            title_response = await app.state.title_generator_llm.ainvoke(title_prompt)
            convo.title = title_response.content.strip().strip('"')
            await db.commit()
    except Exception as e:
        print(f"Could not generate title due to an error (likely rate limit): {e}")

    full_response = ""
    try:
        async for chunk in app.state.agent_executor.astream({"input": question}):
            if "output" in chunk:
                full_response += chunk["output"]
                yield chunk["output"]
    except ResourceExhausted:
        full_response = "I have reached my daily processing limit for this action. My functions will be restored tomorrow."
        yield full_response
    except Exception as e:
        print(f"An unexpected error occurred in the agent stream: {e}")
        full_response = "Sorry, an unexpected error occurred on my end."
        yield full_response
            
    assistant_message = models.Message(role="assistant", content=full_response, conversation_id=conversation_id)
    db.add(assistant_message)
    await db.commit()

@app.post("/conversations/{conversation_id}/stream")
def ask_agent_streaming_endpoint(conversation_id: int, query: Query, x_session_id: Annotated[str, Header()], db: AsyncSession = Depends(get_db)):
    return StreamingResponse(stream_and_save(conversation_id, query.question, x_session_id, db), media_type="text/plain")