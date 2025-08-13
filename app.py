import streamlit as st
import requests
import uuid

# --- Page Configuration ---
st.set_page_config(
    page_title="Agentic RAG Chat", 
    page_icon="🤖", 
    layout="wide"
)

# --- Live Backend API Endpoint ---
API_BASE_URL = "https://ks-agentic-rag-backend.onrender.com"

# =======================================================================================
# --- THE FIX: Using URL Query Params for a unique, persistent session ID ---
# =======================================================================================
# Get the session_id from the URL
try:
    session_id = st.query_params["session_id"]
except:
    # If no session_id, generate a new one and set it in the URL
    session_id = str(uuid.uuid4())
    st.query_params["session_id"] = session_id

# We now include the session_id from the URL in the headers of every API request
headers = {"x-session-id": session_id}

# --- API Helper Functions (now with headers from URL session_id) ---
# (These functions are the same as before, but will now use the correct headers)
def get_conversations():
    try:
        response = requests.get(f"{API_BASE_URL}/conversations/", headers=headers)
        response.raise_for_status(); return response.json()
    except requests.exceptions.RequestException:
        st.error("Could not connect to the backend. Please ensure the server is running and you have refreshed the page."); return []
def create_conversation():
    try:
        response = requests.post(f"{API_BASE_URL}/conversations/", headers=headers)
        response.raise_for_status(); return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to create new conversation: {e}"); return None
def get_conversation_messages(convo_id):
    try:
        response = requests.get(f"{API_BASE_URL}/conversations/{convo_id}", headers=headers)
        response.raise_for_status(); return response.json().get("messages", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to load conversation: {e}"); return []
def delete_conversation(convo_id):
    try:
        response = requests.delete(f"{API_BASE_URL}/conversations/{convo_id}", headers=headers)
        response.raise_for_status(); return True
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to delete conversation: {e}"); return False
def get_response_stream(convo_id, prompt):
    try:
        response = requests.post(f"{API_BASE_URL}/conversations/{convo_id}/stream", headers=headers, json={"question": prompt}, stream=True, timeout=180)
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=None): yield chunk.decode('utf-8')
    except requests.exceptions.RequestException as e: yield f"Error connecting to the agent: {e}"

# --- Session State for UI (messages and current convo ID are still tab-specific) ---
if "messages" not in st.session_state: st.session_state.messages = []
if "current_conversation_id" not in st.session_state: st.session_state.current_conversation_id = None

# --- Sidebar UI ---
with st.sidebar:
    st.title("Conversations")
    if st.button("New Chat", use_container_width=True):
        new_convo = create_conversation()
        if new_convo:
            st.session_state.current_conversation_id = new_convo["id"]
            st.session_state.messages = []
            st.rerun()
    st.write("---")
    conversation_list = get_conversations()
    for convo in conversation_list:
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

# --- Main Chat UI ---
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