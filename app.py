
import streamlit as st
import pandas as pd
from pathlib import Path
import os

# Import your custom modules
from utils import extract_text_from_pdf, parse_csv
from analysis import get_model_and_tokenizer, run_batch_analysis

# --- Page Configuration ---
st.set_page_config(
    page_title="Pathology AI Assistant",
    page_icon="⚕️",
    layout="wide"
)

# --- App State Management ---
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'results_df' not in st.session_state:
    st.session_state.results_df = pd.DataFrame()

# --- UI ---
st.title("⚕️ Pathology AI Diagnostic Assistant")
st.caption("Powered by a local Med-Gemma 4B model")

# --- Model Loading Section ---
if st.session_state.model is None:
    st.info("Model is not loaded. Click the button to load the Med-Gemma model.")
    if st.button("Load Med-Gemma Model"):
        with st.spinner("Loading model... This may take a few minutes and VRAM."):
            try:
                st.session_state.model, st.session_state.tokenizer = get_model_and_tokenizer()
                st.success("Model loaded successfully!")
                st.rerun() # Rerun the script to update the UI
            except Exception as e:
                st.error(f"Failed to load model: {e}")
else:
    st.success("Med-Gemma model is loaded and ready.")

    # --- File Upload Section ---
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV or PDF file",
        type=["csv", "pdf"]
    )

    if uploaded_file is not None:
        # --- Analysis Section ---
        st.header("2. Run Analysis")
        text_column = None
        df = None

        if uploaded_file.type == "text/csv":
            df = parse_csv(uploaded_file)
            st.write("**CSV Preview:**")
            st.dataframe(df.head())
            # Ask user to select the column with diagnosis text
            text_column = st.selectbox("Select the column containing the diagnosis text:", df.columns)
        
        elif uploaded_file.type == "application/pdf":
            with st.spinner("Extracting text from PDF..."):
                pdf_text = extract_text_from_pdf(uploaded_file.getvalue())
                df = pd.DataFrame([{"filename": uploaded_file.name, "diagnosis_text": pdf_text}])
                text_column = "diagnosis_text"
                st.write("**Extracted PDF Text (first 500 chars):**")
                st.text_area("", pdf_text[:500] + "...", height=150)

        if df is not None and text_column and st.button("Run AI Analysis"):
            with st.spinner("Analyzing with Med-Gemma... Please wait."):
                st.session_state.results_df = run_batch_analysis(
                    st.session_state.model, 
                    st.session_state.tokenizer, 
                    df, 
                    text_column
                )
                st.session_state.analysis_complete = True

    # --- Results Section ---
    if st.session_state.analysis_complete:
        st.header("3. Analysis Results")
        st.dataframe(st.session_state.results_df)

        # --- Download Section ---
        st.header("4. Download Results")
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "analysis_results.csv"

        # Convert DataFrame to CSV for download
        csv = st.session_state.results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="pathology_analysis_results.csv",
            mime="text/csv",
        )
