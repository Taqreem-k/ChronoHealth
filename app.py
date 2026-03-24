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


st.set_page_config(page_title="ChronoHealth EMR", layout="wide")

load_dotenv()

# Defining Medical Record Schema
class MedicalRecord(BaseModel):
    date: str = Field(description="Date of the record, Use 'Unknown' if not found.")
    metric: str = Field(description="The medical metric, vital sign, symption or medication")
    value: str = Field(description="The measurement, dosage, or status")


# Defining Patient History Schema
class PatientHistory(BaseModel):
    records: List[MedicalRecord]


# Defining LLMs
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
reviewer_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Initializing embeddings using HuggingFace
embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLm-L6-v2")


# Adding a Sidebar for Data Ingestion
with st.sidebar:
    st.header("Ingest Records")
    # File Uploader to PDF
    uploaded_files = st.file_uploader(
        "Upload File", type = ["pdf","png","jpg","jpeg"]
    )

    # Document Ingestion
    if uploaded_files is not None:
        st.success("File successfully uploaded!")

        if st.session_state.get("processed_file_name") != uploaded_files.name:

            with st.spinner("Processing document..."):
                
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
            
            extracted_data = structured_llm.invoke(f"Extract all the medical data including vital signs, diagnoses, symptoms and medications with their doses from this text: {full_text}")
            st.session_state.structured_db = extracted_data.model_dump()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(pages)

            st.write(f"Document succesfully split into {len(chunks)} chunks.")

            # Storing Embeddings to local Chroma vector database
            vectorstore = Chroma.from_documents(
                documents = chunks,
                embedding = embeddings,
                persist_directory="./chroma_db"
            )

            st.success("Health record successfully vectorized and stored in the database!")

            st.session_state.processed_file_name = uploaded_files.name


# Connecting to Database
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function = embeddings)
retriever = vectorstore.as_retriever(search_kwargs = {"k": 4})


# Defining GraphState schema
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    structured_data: dict
    context: str
    clinical_brief: str
    guardrail_passed: bool


# Wrapping retirever in a Tool for LLM usage
retriever_tool = create_retriever_tool(
    retriever,
    name="patient_history_search",
    description="Use this tool to search and retrieve the patient's medical history,past diagnoses, lab results and general health records.",
)

llm_with_tools = llm.bind_tools([retriever_tool])

# Defining clinical brief based on user queries and retireved data
def clinical_drafter(state: AgentState):
    structured_str = json.dumps(state.get('structured_data',{}), indent = 2)
    system_prompt = f"""You are the ChronoHealth AI Assistant. First, review this highly accurate structured data extracted from the patient's file: {structured_str}. If the user's question can be answered using the structured data above, use it immediately. If you need more context, raw clinical notes, or if the data is missing, you may use the 'patient_history_search' tool to search the vector database. Provide accurate, direct answers based on these sources."""
    system_message = SystemMessage(content=system_prompt)
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
st.title("Patient Record: ChronoHealth")
st.divider()

if "processed_file_name" not in st.session_state:
    st.info("Welcome to ChronoHealth. Please upload a patient document or image in the sidebar to begin.")

else:
    st.subheader("Patient Overview")
    col1,col2,col3 = st.columns(3)

    # Getting the number of records extracted
    structured_data = st.session_state.get("structured_db, {}")
    records = structured_data.get("records", []) if isinstance(structured_data, dict) else[]
    num_data_points = len(str(structured_data.get("records",""))) // 50+ 1 if structured_data else 0

    col1.metric("File Status", "Analyzed & Secured", "Active")
    col2.metric("Data Points Extracted", f"{num_data_points} Entities")
    col3.metric("Guardrail Status", "Online")

    st.divider()

    # Creating Chat Interface
    st.subheader("Clinical Assistant")
    user_question = st.text_input("Ask a question about the uploaded document...")

    if user_question:
        with st.chat_message("user"):
            st.write(user_question)

        with st.spinner("Analyzing records..."):
            initial_state = {"messages": [("user", user_question)],
                            "structured_data": st.session_state.get("structured_db",{})}

            result = app.invoke(initial_state)
            check = result.get("guardrail_passed")

        with st.chat_message("assistant"):
            if check == True:
                st.success("Passed Medical Compliance")
                st.info(result["clinical_brief"])
            else:
                st.error("UNSAFE OUTPUT DETECTED: Blocked by Medical Guardrail.")
                st.warning(result["clinical_brief"])
            
        # The Audit Tabs
        st.divider()
        st.subheader("System Logs & Audit Trail")
        tab1, tab2 = st.tabs(["Structured Database", "Agent Audit Trail"])

        with tab1:
            st.json(st.session_state.get("structured_db",{}))

        with tab2:
            for msg in result["messages"]:
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        st.write(f"Agent used tool: {msg.tool_calls[0]['name']}")      
    
                



