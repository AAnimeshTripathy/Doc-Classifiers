import streamlit as st
import requests
import pandas as pd
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient, ExtractiveSummaryAction
from dotenv import load_dotenv
import os
import re

# Load environment variables
load_dotenv()

# Azure Configuration
DOC_INTEL_ENDPOINT = os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT")
DOC_INTEL_KEY = os.getenv("AZURE_FORM_RECOGNIZER_KEY")
OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT").rstrip('/')  # Remove trailing slash
OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
OPENAI_DEPLOYMENT = os.getenv("OPENAI_DEPLOYMENT")
OPENAI_API_VERSION = "2024-02-15-preview"
AZURE_FORM_RECOGNIZER_ENDPOINT = os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT")
AZURE_FORM_RECOGNIZER_KEY = os.getenv("AZURE_FORM_RECOGNIZER_KEY")
AZURE_LANGUAGE_ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT")
AZURE_LANGUAGE_KEY = os.getenv("AZURE_LANGUAGE_KEY")

def get_document_analysis_client():
    return DocumentAnalysisClient(
        endpoint=DOC_INTEL_ENDPOINT,
        credential=AzureKeyCredential(DOC_INTEL_KEY)
    )

def analyze_document(file_content):
    client = get_document_analysis_client()
    poller = client.begin_analyze_document(
        model_id="prebuilt-layout",
        document=file_content
    )
    return poller.result()

def generate_openai_response(prompt, max_tokens=500):
    try:
        url = f"{OPENAI_ENDPOINT}/openai/deployments/{OPENAI_DEPLOYMENT}/chat/completions"
        params = {"api-version": OPENAI_API_VERSION}
        headers = {
            "Content-Type": "application/json",
            "api-key": OPENAI_KEY
        }
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        response = requests.post(url, headers=headers, params=params, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"OpenAI API error: {str(e)}")
        return None

def generate_summaries(text):
    overall_prompt = f"Provide an overall summary of this document in 3-5 sentences:\n\n{text}"
    section_prompt = f"Provide a section-wise summary of this document, with each section summarized in 1-2 sentences:\n\n{text}"
    
    return (
        generate_openai_response(overall_prompt),
        generate_openai_response(section_prompt)
    )

def classify_document(text):
    prompt = f"Classify this document into categories and provide relevant tags:\n\n{text}"
    return generate_openai_response(prompt)

def convert_table_to_dataframe(table):
    """Convert DocumentTable to pandas DataFrame using row/column indices"""
    max_row = max(cell.row_index for cell in table.cells) if table.cells else 0
    max_col = max(cell.column_index for cell in table.cells) if table.cells else 0
    
    # Initialize empty DataFrame
    df = pd.DataFrame(index=range(max_row+1), columns=range(max_col+1))
    
    # Populate DataFrame
    for cell in table.cells:
        df.iloc[cell.row_index, cell.column_index] = cell.content
    
    # Set headers if column headers exist
    if any(cell.kind == "columnHeader" for cell in table.cells):
        df.columns = df.iloc[0]
        df = df[1:]
    
    return df

def analyze_visual_elements(result):
    """Process tables only (images not supported in current API version)"""
    table_summaries = []
    for i, table in enumerate(result.tables[:3]):  # Process first 3 tables
        try:
            df = convert_table_to_dataframe(table)
            summary = generate_openai_response(f"Summarize this table:\n{df.to_string()}", 100)
            table_summaries.append(f"Table {i+1} Summary:\n{summary}")
        except Exception as e:
            table_summaries.append(f"Error processing table {i+1}: {str(e)}")
    
    return table_summaries

def extract_keywords(text):
    prompt = f"Extract and define key terms from this document in bullet points:\n\n{text}"
    return generate_openai_response(prompt)

def extract_citations_references(text):
    prompt = f"Extract citations, references, and links from this document:\n\n{text}"
    return generate_openai_response(prompt)

def main():
    st.title("📄 Advanced Document Intelligence System")
    
    uploaded_file = st.file_uploader("Upload document (PDF/Image)", type=["pdf", "png", "jpg"])
    
    if uploaded_file:
        with st.spinner("Analyzing document..."):
            try:
                file_content = uploaded_file.read()
                result = analyze_document(file_content)
                text = result.content
                
                with st.expander("Raw Document Analysis", expanded=False):
                    st.json(result.to_dict())

                # Process document components
                overall_summary, section_summary = generate_summaries(text)
                classification = classify_document(text)
                table_summaries = analyze_visual_elements(result)
                keywords = extract_keywords(text)
                citations_references = extract_citations_references(text)

                # Display results
                st.header("Document Analysis Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("1. Summaries")
                    st.markdown(f"**Overall Summary:**\n{overall_summary or 'Not available'}")
                    st.markdown(f"\n**Section-wise Summary:**\n{section_summary or 'Not available'}")
                    
                    st.subheader("2. Classification")
                    st.markdown(classification or "No classification available")

                with col2:
                    st.subheader("3. Tables Analysis")
                    if table_summaries:
                        for summary in table_summaries:
                            st.markdown(f"``````")
                    else:
                        st.write("No tables detected")
                    
                    st.subheader("4. Key Terms")
                    st.markdown(keywords or "No keywords extracted")
                    
                    st.subheader("5. References & Links")
                    st.markdown(citations_references or "No citations found")

            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
                st.stop()

if __name__ == "__main__":
    main()
