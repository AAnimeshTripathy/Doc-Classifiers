import streamlit as st
import requests
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Azure Configuration from .env file
AZURE_FORM_RECOGNIZER_ENDPOINT = os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT")
AZURE_FORM_RECOGNIZER_KEY = os.getenv("AZURE_FORM_RECOGNIZER_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")

# Streamlit App Title
st.title("AI-Powered Document Classification System")

# File Upload Widget
uploaded_file = st.file_uploader("Upload your document (PDF, Image, etc.)", type=["pdf", "png", "jpg"])

if uploaded_file is not None:
    # Step 1: Extract Text Using Azure Form Recognizer (OCR)
    st.info("Processing document...")
    document_analysis_client = DocumentAnalysisClient(
        endpoint=AZURE_FORM_RECOGNIZER_ENDPOINT,
        credential=AzureKeyCredential(AZURE_FORM_RECOGNIZER_KEY)
    )
    
    # Analyze the uploaded file in memory
    poller = document_analysis_client.begin_analyze_document(
        model_id="prebuilt-document", document=uploaded_file.read()
    )
    result = poller.result()
    
    # Extract text from the OCR result
    extracted_text = "\n".join([line.content for page in result.pages for line in page.lines])
    st.subheader("Extracted Text:")
    st.write(extracted_text)

    # Step 2: Classify Text Using Azure OpenAI GPT Models
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_KEY
    }
    data = {
        "messages": [{"role": "system", "content": "You are a document classifier."},
                     {"role": "user", "content": f"Classify and tag this document:\n{extracted_text}"}],
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    response = requests.post(AZURE_OPENAI_ENDPOINT, headers=headers, json=data)
    
    if response.status_code == 200:
        tags_and_classification = response.json()["choices"][0]["message"]["content"]
        st.subheader("Tags and Classification:")
        st.write(tags_and_classification)
    else:
        st.error("Error in classification. Please check your Azure OpenAI configuration.")

# Footer
st.markdown("---")
st.markdown("Powered by Azure AI Services")
