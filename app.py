import streamlit as st

st.title("ChronoHealth")

st.header("About:")

st.subheader("ChronoHealth is an Agentic Personal Health Record System")

uploaded_files = st.file_uploader(
    "Upload images", type = ["pdf"]
)

if uploaded_files is not None:
    st.success("File successfully uploaded!")