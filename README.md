# 🩺 ChronoHealth
**An Agentic, Multi-Modal EMR System with Temporal RAG and Medical Compliance Guardrails.**

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white) ![Gemini](https://img.shields.io/badge/Gemini_2.5_Flash-8E75B2?style=for-the-badge&logo=googlebard&logoColor=white) ![ChromaDB](https://img.shields.io/badge/ChromaDB-FF4F00?style=for-the-badge&logo=chroma&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)

> **ET GenAI Hackathon 2026 Submission** | Track: AI in Healthcare 

---

##  Overview
Modern healthcare is plagued by unstructured data—handwritten prescriptions, messy lab reports, and scattered historical files. Doctors spend immense amounts of time manually parsing this data, leading to burnout and missed clinical trends.

**ChronoHealth** solves this by utilizing a deterministic Multi-Agent State Machine (LangGraph) paired with a multimodal Vision-LLM ingestion layer. It autonomously reads raw medical files, extracts chronological data points into a structured database, and allows clinicians to query patient histories safely, backed by a strict hallucination guardrail.

---

##  Core Features & Workflow

### 1. Clean EMR Dashboard
A modern, intuitive interface designed for healthcare professionals, providing instant visibility into extraction metrics, guardrail status, and patient data.

![EMR Dashboard](assets/dashboard.png)

### 2. Clinical Drafter Agent
Clinicians can ask complex questions about the patient's history. The agent retrieves the relevant temporal data and generates a clean, medically accurate brief.

![Safe Response](assets/safe_response.png)

### 3. Medical Compliance Guardrail
Patient safety is paramount. All drafted responses are intercepted by a secondary LLM Evaluator Node. If the draft contains unverified diagnoses, hallucinations, or unauthorized prescriptions, it is flagged and immediately blocked from the UI.

![Medical Guardrail Block](assets/guardrail_block.png)

### 4. Multi-Modal Vision Ingestion
Bypasses the "digital-only" limitation of standard RAG systems. Uses **Gemini 2.5 Flash** to ingest photos of handwritten doctor's notes, forcing the LLM to extract hard data points (Dates, Metrics, Values) into a structured JSON database.

![Vision Extraction](assets/vision_extraction.png)

### 5. Transparent Audit Trail
Powered by **LangGraph**, the system is fully auditable. Doctors can see exactly which tools the agent used (like `patient_history_search`) to query the ChromaDB vector store, ensuring zero "black box" decisions.

![Agent Audit Trail](assets/audit_trail.png)

---

## System Architecture

Our deterministic state machine ensures that data always flows through our safety checks before reaching the clinician. 

![ChronoHealth Architecture Diagram](assets/architecture.png)

---

##  Quick Start / Installation

**1. Clone the repository**
```bash
git clone [https://github.com/Taqreem-k/ChronoHealth.git](https://github.com/Taqreem-k/ChronoHealth.git)
cd ChronoHealth
```

**2. Set up the environment**
Ensure you have Python 3.10+ installed. Install the dependencies:
```bash
pip install -r requirements.txt
```

**3. Configure API Keys**
Create a `.env` file in the root directory and add your Google Gemini API key:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

**4. Run the EMR Dashboard**
```bash
streamlit run app.py
```

---

##  Tech Stack
* **Orchestration:** LangGraph, LangChain
* **LLMs & Vision:** Google Gemini 2.5 Flash
* **Vector Database:** ChromaDB
* **Data Structuring:** Pydantic
* **Frontend UI:** Streamlit
* **Document Parsing:** PyPDFLoader

---
*Built by M. Taqreem Khan for the ET GenAI Hackathon.*