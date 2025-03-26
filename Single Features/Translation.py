import streamlit as st
import requests
import uuid
import PyPDF2
from io import BytesIO
from dotenv import load_dotenv
import os

# Load environment variables from .env file if needed
load_dotenv()

def translate_text(text, from_lang='en', to_lang='hi'):
    # Your endpoint and subscription key
    endpoint = "https://ai-aihackthonhub282549186415.cognitiveservices.azure.com/translator/text/v3.0/translate"
    subscription_key = "Fj1KPt7grC6bAkNja7daZUstpP8wZTXsV6Zjr2FOxkO7wsBQ5SzQJQQJ99BCACHYHv6XJ3w3AAAAACOGL3Xg"
    
    # Set up the headers with subscription key
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }
    
    # Set up the query parameters
    params = {
        'api-version': '3.0',
        'from': from_lang,
        'to': to_lang
    }
    
    # Set up the request body
    body = [{
        'text': text
    }]
    
    # Make the API request
    try:
        response = requests.post(endpoint, headers=headers, params=params, json=body)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse the JSON response
        result = response.json()
        
        # Extract and return the translated text
        translations = result[0]['translations']
        translated_text = translations[0]['text']
        
        return {
            'original_text': text,
            'translated_text': translated_text,
            'from_language': from_lang,
            'to_language': to_lang,
            'full_response': result
        }
    
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err}")
        return {"error": str(http_err)}
    except Exception as err:
        st.error(f"An error occurred: {err}")
        return {"error": str(err)}

st.title("Document Translator")

st.write("Upload a document (PDF or plain text) to translate its content into Hindi.")

# File uploader accepts PDFs and plain text files.
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])

# Allow user to enter the target language code; default is now "hi" for Hindi.
target_language = st.text_input("Enter target language code (default is 'hi' for Hindi)", "hi")

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    extracted_text = ""
    
    if file_extension == "pdf":
        try:
            # Use PyPDF2 to extract text from PDF
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_text += page_text + "\n"
        except Exception as e:
            st.error("Error reading PDF file: " + str(e))
    elif file_extension == "txt":
        try:
            extracted_text = uploaded_file.read().decode("utf-8")
        except Exception as e:
            st.error("Error reading text file: " + str(e))
    
    if extracted_text:
        st.subheader("Extracted Text:")
        st.write(extracted_text)
        
        st.info("Translating text into Hindi...")
        translation_result = translate_text(extracted_text, from_lang="en", to_lang=target_language)
        
        if "error" in translation_result:
            st.error("Translation failed: " + translation_result["error"])
        else:
            st.subheader("Translated Text:")
            st.write(translation_result["translated_text"])
