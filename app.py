# # app.py
# import os, faiss, pickle, numpy as np, streamlit as st
# from sentence_transformers import SentenceTransformer
# import google.generativeai as genai
# from api_manager import GeminiKeyManager

# # Load FAISS + metadata
# INDEX_PATH = "faiss_index/index.faiss"
# META_PATH = "faiss_index/meta.pkl"
# MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# @st.cache_resource
# def load_index():
#     index = faiss.read_index(INDEX_PATH)
#     with open(META_PATH, "rb") as f:
#         meta = pickle.load(f)
#     return index, meta

# index, meta = load_index()
# api_manager = GeminiKeyManager()

# # Streamlit UI
# st.set_page_config(page_title="Bharathiar University Chatbot", layout="centered")
# st.title("🎓 Bharathiar University Inquiry Chatbot")
# st.caption("Ask questions about your Department, Courses, or Admission")

# query = st.text_input("Ask your question:")
# top_k = st.slider("Documents to retrieve", 1, 8, 3)

# if st.button("Ask"):
#     if not query.strip():
#         st.warning("Please enter a question.")
#     else:
#         q_vec = MODEL.encode([query], convert_to_numpy=True)
#         D, I = index.search(q_vec, top_k)
#         hits = [meta[i] for i in I[0]]

#         # Build context
#         context = "\n\n".join([f"Source: {h['source']}\n{h['text']}" for h in hits])
#         prompt = f"""
# You are an assistant for the Bharathiar University Department of Computer Applications.
# Use the context below (from university documents) to answer the student's question.
# If not found in context, say "I don't know based on the available university documents."

# Context:
# {context}

# Question: {query}

# Answer politely and clearly.
# """

#         # Rotate Gemini key
#         key = api_manager.get_key()
#         genai.configure(api_key=key)
#         try:
#             model = genai.GenerativeModel("gemini-2.5-flash")  # updated model
#             response = model.generate_content(prompt)  
#             st.success("💬 Answer:")
#             st.write(response.text)
#         except Exception as e:
#             st.error(f"Gemini Error: {e}")

#         with st.expander("📄 Sources used"):
#             for h in hits:
#                 st.write(f"- {h['source']}")





# app.py
import os
import faiss
import pickle
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from api_manager import GeminiKeyManager

# ------------------------
# SETTINGS AND PATHS
# ------------------------
INDEX_PATH = "faiss_index/index.faiss"
META_PATH = "faiss_index/meta.pkl"
MODEL = SentenceTransformer("all-MiniLM-L6-v2")
TOP_K_DEFAULT = 3

# Initialize session_state
if "history" not in st.session_state:
    st.session_state.history = []

# ------------------------
# LOAD FAISS INDEX
# ------------------------
@st.cache_resource
def load_index():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        st.error("❌ FAISS index or meta not found. Run ingest.py first.")
        return None, None
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return index, meta

index, meta = load_index()
api_manager = GeminiKeyManager()

# ------------------------
# STREAMLIT UI
# ------------------------
st.set_page_config(page_title="Bharathiar University Chatbot", layout="wide")

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/5/5b/Bharathiar_University_Logo.png", width=150)
    st.title("🎓 BU Chatbot")
    st.markdown("""
    **Department of Computer Applications**

    Ask anything about:
    - Courses & Syllabus
    - Admission & Eligibility
    - Departments & Faculty
    - Campus & Facilities
    """)
    st.markdown("---")
    st.slider("Documents to retrieve", 1, 8, TOP_K_DEFAULT, key="top_k")

# Main chat window
st.title("Bharathiar University Inquiry Chatbot")
st.markdown("💬 Ask questions about your department, courses, or admission.")

# ------------------------
# USER INPUT FORM
# ------------------------
with st.form(key="chat_form", clear_on_submit=True):
    query = st.text_input("Type your question here:")
    submit = st.form_submit_button("Send")

# ------------------------
# PROCESS CHAT
# ------------------------
if submit and query.strip():
    st.session_state.history.append({"role": "user", "message": query})

    if index is not None:
        # Encode query and search FAISS
        q_vec = MODEL.encode([query], convert_to_numpy=True)
        D, I = index.search(q_vec, st.session_state.top_k)
        hits = [meta[i] for i in I[0]]

        # Build context
        context = "\n\n".join([f"Source: {h['source']}\n{h['text']}" for h in hits])
        prompt = f"""
You are an assistant for the Bharathiar University Department of Computer Applications.
Use the context below (from university documents) to answer the student's question.
If not found in context, say "I don't know based on the available university documents."

Context:
{context}

Question: {query}

Answer politely and clearly.
"""

        # Rotate Gemini key
        key = api_manager.get_key()
        genai.configure(api_key=key)

        with st.spinner("💡 Thinking..."):
            try:
                model_g = genai.GenerativeModel("gemini-2.5-flash")
                response = model_g.generate_content(prompt)
                answer = response.text
            except Exception as e:
                answer = f"Gemini Error: {e}"

        st.session_state.history.append({"role": "bot", "message": answer})

# ------------------------
# DISPLAY CHAT HISTORY WITH STYLE
# ------------------------
def render_message(message, role="user"):
    if role == "user":
        st.markdown(
            f"""
            <div style="
                display: flex; justify-content: flex-end; margin:5px;
                ">
                <div style="
                    background-color: #DCF8C6; padding: 12px; border-radius: 15px;
                    max-width:70%; font-size:16px; line-height:1.4;
                ">
                    {message}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="
                display: flex; justify-content: flex-start; margin:5px;
                ">
                <div style="
                    background-color: #F1F0F0; padding: 12px; border-radius: 15px;
                    max-width:70%; font-size:16px; line-height:1.4;
                ">
                    {message}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

for chat in st.session_state.history:
    render_message(chat["message"], chat["role"])

# ------------------------
# SOURCES EXPANDER
# ------------------------
if index is not None and st.session_state.history:
    with st.expander("📄 Sources used in last answer"):
        last_hits = [meta[i] for i in I[0]]
        for h in last_hits:
            st.write(f"- {h['source']}")
