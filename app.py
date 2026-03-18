import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

st.title("ChronoHealth")

st.header("About:")

st.subheader("ChronoHealth is an Agentic Personal Health Record System")

uploaded_files = st.file_uploader(
    "Upload PDF", type = ["pdf"]
)

if uploaded_files is not None:
    st.success("File successfully uploaded!")

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_files.getvalue())
    
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(pages)

    st.write(f"Document succesfully split into {len(chunks)} chunks.")