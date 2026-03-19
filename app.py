import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import TypedDict
from langchain_core.tools.retriever import create_retriever_tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()


st.title("ChronoHealth")
st.header("About:")
st.subheader("ChronoHealth is an Agentic Personal Health Record System")

# File Uploader to PDF
uploaded_files = st.file_uploader(
    "Upload PDF", type = ["pdf"]
)

# Document Ingestion
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

    # Storing Embeddings to local Chroma vector database
    vectorstore = Chroma.from_documents(
        documents = chunks,
        embedding = embeddings,
        persist_directory="./chroma_db"
    )

    st.success("Health record successfully vectorized and sotred in the database!")

    # Defining GraphState schema
    class AgentState(TypedDict):
        query: str
        context: str
        clinical_brief: str
        guardrail_passed: bool
        raw_message: any
    
    retriever = vectorstore.as_retriever()

    # Wrapping retirever in a Tool for LLM usage
    retriever_tool = create_retriever_tool(
        retriever,
        name="patient_history_search",
        description="Use this tool to search and retrieve the patient's medical history,past diagnoses, lab results and general health records.",
    )

    llm = ChatGroq(model_name = "llama-3.1-8b-instant")
    llm_with_tools = llm.bind_tools([retriever_tool])


    # Defining clinical brief based on user queries and retireved data
    def clinical_drafter(state: AgentState):
        user_query = state["query"]
        response = llm_with_tools.invoke(user_query)

        return{
            "clinical_brief": response.content,
            "raw_message": response,
        }
    
    tool_executer = ToolNode(tools=[retriever_tool])

    # Initializing Graph
    workflow = StateGraph(AgentState)

    workflow.add_node("drafter", clinical_drafter)
    workflow.add_node("tools", tool_executer)

    workflow.set_entry_point("drafter")


