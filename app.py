import streamlit as st
import base64
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import TypedDict, Annotated, List
from langchain_core.tools.retriever import create_retriever_tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import json

load_dotenv()

# Defining Medical Record Schema
class MedicalRecord(BaseModel):
    date: str = Field(description="Date of the record, Use 'Unknown' if not found.")
    metric: str = Field(description="The medical metric, vital sign, symption or medication")
    value: str = Field(description="The measurement, dosage, or status")

# Defining Patient History Schema
class PatientHistory(BaseModel):
    records: List[MedicalRecord]


st.title("ChronoHealth")
st.header("About:")
st.subheader("ChronoHealth is an Agentic Personal Health Record System")


# Defining LLMs
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
reviewer_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# File Uploader to PDF
uploaded_files = st.file_uploader(
    "Upload File", type = ["pdf","png","jpg","jpeg"]
)

# Document Ingestion
if uploaded_files is not None:
    st.success("File successfully uploaded!")

    if uploaded_files.name.endswith(".pdf"):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_files.getvalue())
        
        loader = PyPDFLoader("temp.pdf")
        pages = loader.load()

    else:
        image_data = base64.b64encode(uploaded_files.getvalue()).decode("utf-8")
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Transcribe this medical docuemtn exactly as written. Do not add any extra commentary. Just extract the text and medica values."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ]
        )
        vision_response = llm.invoke([message])
        pages = [Document(page_content=vision_response.content)]

    full_text = "\n".join([page.page_content for page in pages])
    structured_llm = llm.with_structured_output(PatientHistory)
    extracted_data = structured_llm.invoke(f"Extract all the medical data from this text: {full_text}")
    
    with st.expander("View Structured Databse Entry"):
        st.json(extracted_data.dict())

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

    st.success("Health record successfully vectorized and stored in the database!")

    # Defining GraphState schema
    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]
        context: str
        clinical_brief: str
        guardrail_passed: bool

    retriever = vectorstore.as_retriever(search_kwargs = {"k": 2})
    

    # Wrapping retirever in a Tool for LLM usage
    retriever_tool = create_retriever_tool(
        retriever,
        name="patient_history_search",
        description="Use this tool to search and retrieve the patient's medical history,past diagnoses, lab results and general health records.",
    )

    llm_with_tools = llm.bind_tools([retriever_tool])

    # Defining clinical brief based on user queries and retireved data
    def clinical_drafter(state: AgentState):
        system_message = SystemMessage(content="You are the ChronoHealth AI Assistant. You Must use the 'patient_history_search' tool to answer any questions about the user's uploaded document. Never answer from general knowledge.")
        messages_to_pass = [system_message] + state["messages"]
        response = llm_with_tools.invoke(messages_to_pass)

        if isinstance(response.content, list):
            clean_text = response.content[0]["text"]
        else:
            clean_text = response.content

        return{
            "clinical_brief": clean_text,
            "messages": [response],
        }

    # Defining a guardrail checker to prevent hallucinations
    def compliance_checker(state: AgentState):
        extracted_brief = state["clinical_brief"]
        new_system_message = f"You are a medical compliance reviewer. Review this clinical brief. If it contains dangerous assumptions or diagnoses without evidence, reply ONLY with 'UNSAFE'. Otherwise, reply 'SAFE'. Brief:{extracted_brief}"
        new_response = reviewer_llm.invoke(new_system_message)

        if "UNSAFE" in new_response.content:
            return {"guardrail_passed": False}
        else:
            return {"guardrail_passed": True}

    # Defining a routing function
    def route_after_drafter(state: AgentState):
        last_message= state["messages"][-1]

        if last_message.tool_calls:
            return "tools"
        else:
            return "guardrail"


    tool_executer = ToolNode(tools=[retriever_tool])

    # Initializing Graph to add nodes and edges
    workflow = StateGraph(AgentState)

    workflow.add_node("drafter", clinical_drafter)
    workflow.add_node("tools", tool_executer)
    workflow.add_node("guardrail", compliance_checker)

    workflow.set_entry_point("drafter")

    workflow.add_conditional_edges("drafter",route_after_drafter)
    workflow.add_edge("tools","drafter")
    workflow.add_edge("guardrail", END)

    app = workflow.compile()

    # Definining user query
    st.subheader("Query Patient Record")
    user_question = st.text_input("Ask a question about the uploaded document: ")

    if user_question:
        with st.spinner("Agent is searching records and drafting brief..."):
            initial_state = {"messages": [("user", user_question)]}

            result = app.invoke(initial_state)
            check = result.get("guardrail_passed")

            if check == True:
                st.success("Passed Medical Compliance")
                with st.expander("View Agent Audit Trail"):
                    for msg in result["messages"]:
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            st.write(f"Agent used tool: {msg.tool_calls[0]['name']}")
                st.write("Final Clinical Brief")
                st.write(result["clinical_brief"])
            else:
                st.error("UNSAFE OUTPUT DETECTED: Blocked by Medical Guardrail.")
                with st.expander("View Blocked Draft"):
                    st.write(result["clinical_brief"])
                        
        
                    



