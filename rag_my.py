import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
import time # Import time for demonstration

key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=key)

gemini_model = genai.GenerativeModel('gemini-2.0-flash')

@st.cache_data(show_spinner='Loading Embedding Model...')
def load_embedding_model():
    # Add a time.sleep() to simulate a longer loading time
    # This will make the spinner more noticeable if it's actually loading
    time.sleep(3)
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.header('RAG Assistant :blue[Using Embedding & Gemini LLM]')
st.subheader('Your Intelligent Document Assistant!')

st.write('Done')

# Call the cached function
embedding_model = load_embedding_model()
st.write("Embedding model loaded!") # Confirmation message


uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])


if uploaded_file:
    st.write('Uploaded successfully!')

    pdf = PdfReader(uploaded_file)
    raw_text = ''

    for page in pdf.pages:
        raw_text += page.extract_text()

    st.write('Extracted successfully!')

# Add a simple button to trigger a rerun without changing the cached function's code
if st.button("Rerun App (should use cache)"):
    st.write("App reran. Check console/spinner for caching behavior.")