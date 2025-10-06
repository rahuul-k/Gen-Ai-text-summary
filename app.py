import streamlit as st
from summarizer import get_hf_summarizer, get_openai_summarizer
from utils import read_txt, read_pdf, chunk_text
import os

st.set_page_config(page_title="GenAI Text Summarizer", layout="centered")

st.title("GenAI Text Summarizer")
st.markdown("Upload a text or PDF file, or paste text. Choose backend and summary length.")

col1, col2 = st.columns([3,1])

with col1:
    uploaded = st.file_uploader("Upload a .txt or .pdf file", type=['txt','pdf'])
    text_input = st.text_area("Or paste text here", height=200)

with col2:
    backend = st.selectbox("Backend", ['huggingface', 'openai'])
    length = st.radio("Summary length", ['short','medium','long'])
    hf_model = st.text_input("HF model name (for huggingface)", value="facebook/bart-large-cnn")
    openai_model = st.text_input("OpenAI model (for OpenAI)", value="gpt-4o-mini")
    run_btn = st.button("Summarize")

content = ""
if uploaded:
    if uploaded.type == 'text/plain' or uploaded.name.endswith('.txt'):
        content = read_txt(uploaded)
    elif uploaded.type == 'application/pdf' or uploaded.name.endswith('.pdf'):
        content = read_pdf(uploaded)
    else:
        st.warning('Unsupported file type. Please upload .txt or .pdf')

if text_input and not content:
    content = text_input

if run_btn:
    if not content:
        st.warning('Please upload or paste some text first.')
    else:
        chunks = chunk_text(content, max_chars=3000)
        st.info(f"Text split into {len(chunks)} chunk(s)")
        try:
            if backend == 'huggingface':
                summarizer = get_hf_summarizer(model_name=hf_model)
                if length == 'short':
                    max_len, min_len = 80, 20
                elif length == 'medium':
                    max_len, min_len = 150, 40
                else:
                    max_len, min_len = 300, 80
                summary = summarizer.summarize(chunks, max_length=max_len, min_length=min_len)
            else:
                summarizer = get_openai_summarizer(model=openai_model)
                summary = summarizer.summarize(chunks, length=length)
            st.subheader('Summary')
            st.write(summary)
            st.download_button('Download summary', data=summary, file_name='summary.txt')
        except Exception as e:
            st.error(f"Error while summarizing: {e}")
