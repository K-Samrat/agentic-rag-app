import streamlit as st
import requests

st.set_page_config(page_title="Agentic RAG Chat", page_icon="🤖", layout="wide")
# API_BASE_URL = "http://127.0.0.1:8000"
API_BASE_URL = "https://ks-agentic-rag-backend.onrender.com"
def get_conversations():
    try:
        response = requests.get(f"{API_BASE_URL}/conversations/")
        response.raise_for_status(); return response.json()
    except requests.exceptions.RequestException: return []
def create_conversation():
    try:
        response = requests.post(f"{API_BASE_URL}/conversations/")
        response.raise_for_status(); return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to create new conversation: {e}"); return None
def get_conversation_messages(convo_id):
    try:
        response = requests.get(f"{API_BASE_URL}/conversations/{convo_id}")
        response.raise_for_status(); return response.json().get("messages", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to load conversation: {e}"); return []
def delete_conversation(convo_id):
    try:
        response = requests.delete(f"{API_BASE_URL}/conversations/{convo_id}")
        response.raise_for_status(); return True
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to delete conversation: {e}"); return False
def get_response_stream(convo_id, prompt):
    try:
        response = requests.post(f"{API_BASE_URL}/conversations/{convo_id}/stream", json={"question": prompt}, stream=True, timeout=180)
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=None): yield chunk.decode('utf-8')
    except requests.exceptions.RequestException as e: yield f"Error connecting to the agent: {e}"
if "messages" not in st.session_state: st.session_state.messages = []
if "current_conversation_id" not in st.session_state: st.session_state.current_conversation_id = None
if "conversation_list" not in st.session_state: st.session_state.conversation_list = []
with st.sidebar:
    st.title("Conversations")
    if st.button("New Chat", use_container_width=True):
        new_convo = create_conversation()
        if new_convo:
            st.session_state.current_conversation_id = new_convo["id"]
            st.session_state.messages = []
            st.rerun()
    st.write("---")
    st.session_state.conversation_list = get_conversations()
    for convo in st.session_state.conversation_list:
        convo_id = convo["id"]
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(convo["title"], key=f"convo_{convo_id}", use_container_width=True):
                st.session_state.current_conversation_id = convo_id
                st.session_state.messages = get_conversation_messages(convo_id)
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"delete_{convo_id}", use_container_width=True):
                if delete_conversation(convo_id):
                    if st.session_state.current_conversation_id == convo_id:
                        st.session_state.current_conversation_id = None
                        st.session_state.messages = []
                    st.rerun()
st.title("🤖 Agentic RAG Final Version")
st.caption("All features enabled: Streaming, History, Titling, Delete & Multi-Tool Agent")
if st.session_state.current_conversation_id is None:
    st.info("Click 'New Chat' in the sidebar to begin.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            convo_id = st.session_state.current_conversation_id
            full_response = st.write_stream(get_response_stream(convo_id, prompt))
        st.session_state.messages.append({"role": "assistant", "content": full_response})