import streamlit as st
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient, ExtractiveSummaryAction
from dotenv import load_dotenv
import os
import re

# Load environment variables
load_dotenv()

# Azure Configuration
AZURE_FORM_RECOGNIZER_ENDPOINT = os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT")
AZURE_FORM_RECOGNIZER_KEY = os.getenv("AZURE_FORM_RECOGNIZER_KEY")
AZURE_LANGUAGE_ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT")
AZURE_LANGUAGE_KEY = os.getenv("AZURE_LANGUAGE_KEY")

# Streamlit App Configuration
st.set_page_config(page_title="Document Classification & Summarization", layout="wide")
st.title("📄 AI-Powered Document Processing System")

class DocumentProcessor:
    def __init__(self):
        self.form_recognizer_client = DocumentAnalysisClient(
            endpoint=AZURE_FORM_RECOGNIZER_ENDPOINT,
            credential=AzureKeyCredential(AZURE_FORM_RECOGNIZER_KEY)
        )
        self.language_client = TextAnalyticsClient(
            endpoint=AZURE_LANGUAGE_ENDPOINT,
            credential=AzureKeyCredential(AZURE_LANGUAGE_KEY)
        )

    def extract_text(self, uploaded_file):
        """Extract text from the uploaded document using Form Recognizer."""
        poller = self.form_recognizer_client.begin_analyze_document(
            model_id="prebuilt-document", document=uploaded_file.read()
        )
        result = poller.result()

        extracted_text = "\n".join([line.content for page in result.pages for line in page.lines])
        return extracted_text

    def segment_sections(self, text):
        """
        Automatically segments document text into sections using pattern detection.
        Sections are identified based on headings (e.g., bold, capitalized, or underlined text).
        """
        sections = {}
        lines = text.split("\n")
        current_section = "General"
        sections[current_section] = []

        for line in lines:
            # Detect section titles (heuristics: capitalized words, colon presence, or numbering)
            if re.match(r"^\s*([A-Z][A-Za-z\s-]+:|\d+\.\s+[A-Z][A-Za-z\s-]+)", line):
                current_section = line.strip().rstrip(":")  # Remove trailing colon if present
                sections[current_section] = []
            else:
                sections[current_section].append(line.strip())

        # Convert list of lines into paragraph text per section
        return {section: " ".join(content).strip() for section, content in sections.items() if content}

    def generate_extractive_summary(self, text, max_sentences=3):
        """Extracts key sentences using Azure's extractive summarization."""
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
        return " ".join(summary_sentences)  # Returning as a single paragraph

def main():
    processor = DocumentProcessor()

    uploaded_file = st.file_uploader("Upload your document (PDF, DOCX)", type=["pdf", "docx"])

    if uploaded_file:
        with st.spinner("Processing document..."):
            extracted_text = processor.extract_text(uploaded_file)

            if len(extracted_text) < 50:
                st.error("Insufficient text extracted - check document quality.")
                return

            # Display extracted text
            # st.subheader("📄 Extracted Text")
            # st.write(extracted_text[:1000] + ("..." if len(extracted_text) > 10000 else ""))

            # Identify Sections
            sections = processor.segment_sections(extracted_text)

            # Generate and Display Section-wise Summaries
            st.subheader("📝 Section-wise Summarization")
            for section, content in sections.items():
                st.markdown(f"### {section}")
                summary = processor.generate_extractive_summary(content)
                st.write(summary if summary else "No significant content to summarize.")

if __name__ == "__main__":
    main()