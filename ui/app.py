import streamlit as st
import requests
import json

# Configure page
st.set_page_config(page_title="Nutrition AI Assistant", layout="wide")
st.title("🥗 Nutrition AI Assistant")

# API endpoint
API_URL = "http://localhost:8000"

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
col1, col2 = st.columns([4, 1])
with col1:
    user_query = st.text_input("Ask about nutrition recommendations:")
with col2:
    submit_btn = st.button("Send", type="primary")

# Process query
if submit_btn and user_query:
    with st.spinner("Processing..."):
        try:
            response = requests.post(f"{API_URL}/chat", params={"query": user_query})
            result = response.json()
            
            if "error" in result:
                st.error(result["error"])
            else:
                st.session_state.chat_history.append({
                    "query": user_query,
                    "result": result
                })
        except Exception as e:
            st.error(f"Connection error: {e}")

# Display chat history
if st.session_state.chat_history:
    st.divider()
    for i, item in enumerate(reversed(st.session_state.chat_history)):
        with st.container(border=True):
            st.write(f"**Your question:** {item['query']}")
            st.write(f"**Recommendation:** {item['result']['recommendation']}")
            with st.expander("View constraints"):
                st.json(item['result']['constraints'])