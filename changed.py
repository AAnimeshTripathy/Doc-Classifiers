import streamlit as st
import requests
import pandas as pd
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient, ExtractiveSummaryAction
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
import fitz  # PyMuPDF for PDF image extraction
from PIL import Image
from dotenv import load_dotenv
import os
import re
import io
from PIL import Image

# Load environment variables
load_dotenv()

# Azure Configuration
DOC_INTEL_ENDPOINT = os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT")
DOC_INTEL_KEY = os.getenv("AZURE_FORM_RECOGNIZER_KEY")
OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT").rstrip('/') # Remove trailing slash
OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
OPENAI_DEPLOYMENT = os.getenv("OPENAI_DEPLOYMENT")
OPENAI_API_VERSION = "2024-02-15-preview"
AZURE_LANGUAGE_ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT")
AZURE_LANGUAGE_KEY = os.getenv("AZURE_LANGUAGE_KEY")
AZURE_COMPUTER_VISION_ENDPOINT = os.getenv("AZURE_COMPUTER_VISION_ENDPOINT")
AZURE_COMPUTER_VISION_KEY = os.getenv("AZURE_COMPUTER_VISION_KEY")

# Streamlit App Configuration
st.set_page_config(page_title="Document Classification & Summarization", layout="wide")
st.title("📄 AI-Powered Document Processing System")


class DocumentProcessor:
    def __init__(self):
        self.form_recognizer_client = DocumentAnalysisClient(
            endpoint=DOC_INTEL_ENDPOINT,
            credential=AzureKeyCredential(DOC_INTEL_KEY)
        )
        self.language_client = TextAnalyticsClient(
            endpoint=AZURE_LANGUAGE_ENDPOINT,
            credential=AzureKeyCredential(AZURE_LANGUAGE_KEY)
        )
        self.vision_client = ComputerVisionClient(
            AZURE_COMPUTER_VISION_ENDPOINT,
            CognitiveServicesCredentials(AZURE_COMPUTER_VISION_KEY)
        )

    def extract_text(self, uploaded_file):
        poller = self.form_recognizer_client.begin_analyze_document(
            model_id="prebuilt-layout",
            document=uploaded_file.read()
        )
        result = poller.result()
        return result, "\n".join([line.content for page in result.pages for line in page.lines])

    def segment_sections(self, text):
        sections = {}
        lines = text.split("\n")
        current_section = "General"
        sections[current_section] = []
        for line in lines:
            if re.match(r"^\s*([A-Z][A-Za-z\s-]+:|\d+\.\s+[A-Z][A-Za-z\s-]+)", line):
                current_section = line.strip().rstrip(":")
                sections[current_section] = []
            else:
                sections[current_section].append(line.strip())
        return {section: " ".join(content).strip() for section, content in sections.items() if content}

    def generate_extractive_summary(self, text, max_sentences=3):
        poller = self.language_client.begin_analyze_actions(
            documents=[{"id": "1", "language": "en", "text": text}],
            actions=[ExtractiveSummaryAction(max_sentence_count=max_sentences)]
        )
        document_results = poller.result()
        summary_sentences = []
        for result in document_results:
            extract_summary_result = result[0]
            if not extract_summary_result.is_error:
                summary_sentences = [sentence.text for sentence in extract_summary_result.sentences]
        return " ".join(summary_sentences)

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
    return generate_openai_response(overall_prompt), generate_openai_response(section_prompt)

def classify_document(text):
    prompt = f"Classify this document into categories and provide relevant tags:\n\n{text}"
    return generate_openai_response(prompt)

def convert_table_to_dataframe(table):
    max_row = max(cell.row_index for cell in table.cells) if table.cells else 0
    max_col = max(cell.column_index for cell in table.cells) if table.cells else 0
    df = pd.DataFrame(index=range(max_row+1), columns=range(max_col+1))
    for cell in table.cells:
        df.iloc[cell.row_index, cell.column_index] = cell.content
    if any(cell.kind == "columnHeader" for cell in table.cells):
        df.columns = df.iloc[0]
        df = df[1:]
    return df

def analyze_visual_elements(result):
    table_summaries = []
    table_dataframes = []
    for i, table in enumerate(result.tables[:3]):
        try:
            df = convert_table_to_dataframe(table)
            table_dataframes.append(df)
            summary = generate_openai_response(f"Summarize this table:\n{df.to_string()}", 100)
            table_summaries.append(f"Table {i+1} Summary:\n{summary}")
        except Exception as e:
            table_summaries.append(f"Error processing table {i+1}: {str(e)}")
    return table_summaries, table_dataframes

def extract_keywords(text):
    prompt = f"Extract and define key terms from this document in bullet points:\n\n{text}"
    return generate_openai_response(prompt)

def extract_citations_references(text):
    prompt = f"Extract citations, references, and links from this document:\n\n{text}"
    return generate_openai_response(prompt)


def main():
    processor = DocumentProcessor()
    uploaded_file = st.file_uploader("Upload your document (PDF, DOCX)", type=["pdf", "docx"])

    if uploaded_file:
        with st.spinner("Processing document..."):
            try:
                result, extracted_text = processor.extract_text(uploaded_file)
                
                if len(extracted_text) < 50:
                    st.error("Insufficient text extracted - check document quality.")
                    return

                sections = processor.segment_sections(extracted_text)
                
                st.subheader("📝 Section-wise Summarization (Extractive)")
                for section, content in sections.items():
                    st.markdown(f"### {section}")
                    summary = processor.generate_extractive_summary(content)
                    st.write(summary if summary else "No significant content to summarize.")

                with st.expander("Raw Document Analysis", expanded=False):
                    st.json(result.to_dict())

                overall_summary, section_summary = generate_summaries(extracted_text)
                classification = classify_document(extracted_text)
                table_summaries, table_dataframes = analyze_visual_elements(result)
                keywords = extract_keywords(extracted_text)
                citations_references = extract_citations_references(extracted_text)

                st.header("Additional Document Analysis Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("1. Overall Summary")
                    st.markdown(overall_summary or "Not available")
                    st.subheader("2. Classification")
                    st.markdown(classification or "No classification available")
                with col2:
                    st.subheader("3. Tables Analysis")
                    if table_dataframes:
                        for i, df in enumerate(table_dataframes):
                            st.write(f"Table {i+1}")
                            st.dataframe(df)
                            st.markdown(table_summaries[i])
                    else:
                        st.write("No tables detected")
                    st.subheader("4. Key Terms")
                    st.markdown(keywords or "No keywords extracted")
                    st.subheader("5. References & Links")
                    st.markdown(citations_references or "No citations found")

                st.subheader("Images")
                for page in result.pages:
                    for line in page.lines:
                        # Check if the line contains image data
                        if hasattr(line, 'content') and line.content:
                            try:
                                # Attempt to open the content as an image
                                image = Image.open(io.BytesIO(line.content))
                                st.image(image, caption=f"Image on page {page.page_number}")
                            except Exception as e:
                                # If it's not an image, continue to the next line
                                continue



            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
                st.stop()

if __name__ == "__main__":
    main()
