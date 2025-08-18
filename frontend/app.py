import streamlit as st
import requests

st.set_page_config(page_title="Medical Q&A System", page_icon="🏥", layout="centered")

st.title("🏥 Medical Q&A System")
st.write("Ask any medical question and get answers from our knowledge base.")

# Input
user_input = st.text_input("Enter your medical question:")

if st.button("Ask"):
    if user_input.strip() == "":
        st.warning("Please enter a question.")
    else:
        try:
            response = requests.post(
                "http://127.0.0.1:8000/ask",  # FastAPI endpoint
                json={"question": user_input}
            )
            if response.status_code == 200:
                answer = response.json().get("answer", "No answer found.")
                st.success(answer)
            else:
                st.error("Error fetching answer. Please try again.")
        except Exception as e:
            st.error(f"Backend not running: {e}")

