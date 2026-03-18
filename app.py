import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

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

    embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLm-L6-v2")

    vectorstore = Chroma.from_documents(
        documents = chunks,
        embedding = embeddings,
        persist_directory="./chroma_db"
    )

    st.success("Health record successfully vectorized and sotred in the database!")