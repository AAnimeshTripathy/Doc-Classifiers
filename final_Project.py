import streamlit as st
import requests
import pandas as pd
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient, ExtractiveSummaryAction
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
import fitz
from PIL import Image
from dotenv import load_dotenv
import os
import re
import io

load_dotenv()

DOC_INTEL_ENDPOINT = os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT")
DOC_INTEL_KEY = os.getenv("AZURE_FORM_RECOGNIZER_KEY")
OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT").rstrip('/')
OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
OPENAI_DEPLOYMENT = os.getenv("OPENAI_DEPLOYMENT")
OPENAI_API_VERSION = "2024-02-15-preview"
AZURE_LANGUAGE_ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT")
AZURE_LANGUAGE_KEY = os.getenv("AZURE_LANGUAGE_KEY")
AZURE_COMPUTER_VISION_ENDPOINT = os.getenv("AZURE_COMPUTER_VISION_ENDPOINT")
AZURE_COMPUTER_VISION_KEY = os.getenv("AZURE_COMPUTER_VISION_KEY")

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

def get_image_name(analysis_result, index):
    if analysis_result.description and analysis_result.description.captions:
        caption = analysis_result.description.captions[0].text.strip()
        generic_terms = ["diagram", "chart", "graph", "image", "picture"]
        if caption.lower() in generic_terms:
            if analysis_result.objects:
                objects = ", ".join([obj.object_property for obj in analysis_result.objects])
                return f"{caption} of {objects}"
            else:
                return caption
        else:
            return caption
    elif analysis_result.objects:
        return ", ".join([obj.object_property for obj in analysis_result.objects])
    else:
        return f"Image {index}"

def preprocess_pdf(file_bytes):
    pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
    new_pdf = fitz.open()
    for page in pdf_doc:
        new_pdf.insert_pdf(pdf_doc, from_page=page.number, to_page=page.number)
    output_stream = io.BytesIO()
    new_pdf.save(output_stream)
    return output_stream.getvalue()

def main():
    processor = DocumentProcessor()
    uploaded_file = st.file_uploader("Upload your document (PDF, DOCX, PNG, JPG)", type=["pdf", "docx", "png", "jpg"])

    if uploaded_file:
        with st.spinner("Processing document..."):
            try:
                file_bytes = uploaded_file.read()
                if uploaded_file.type == "application/pdf":
                    file_bytes = preprocess_pdf(file_bytes)
                
                result = processor.form_recognizer_client.begin_analyze_document(
                    model_id="prebuilt-layout",
                    document=file_bytes
                ).result()
                extracted_text = "\n".join([line.content for page in result.pages for line in page.lines])

                # if len(extracted_text) < 50:
                #     st.error("Insufficient text extracted - check document quality.")
                #     return

                sections = processor.segment_sections(extracted_text)
                overall_summary, section_summary = generate_summaries(extracted_text)
                classification = classify_document(extracted_text)
                table_summaries, table_dataframes = analyze_visual_elements(result)
                keywords = extract_keywords(extracted_text)
                citations_references = extract_citations_references(extracted_text)

                st.header("Document Analysis Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("1. Overall Summary")
                    st.markdown(overall_summary or "Not available")
                    st.subheader("2. Classification")
                    st.markdown(classification or "No classification available")
                    st.subheader("3. Section-wise Summary")
                    st.markdown(section_summary or "Not available")
                with col2:
                    st.subheader("4. Tables Analysis")
                    if table_dataframes:
                        for i, df in enumerate(table_dataframes):
                            st.write(f"Table {i+1}")
                            st.dataframe(df)
                            st.markdown(table_summaries[i])
                    else:
                        st.write("No tables detected")
                    st.subheader("5. Key Terms")
                    st.markdown(keywords or "No keywords extracted")
                    st.subheader("6. References & Links")
                    st.markdown(citations_references or "No citations found")

                st.subheader("Images")
                image_objects = []
                if uploaded_file.type == "application/pdf":
                    pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
                    for page_index in range(pdf_doc.page_count):
                        page = pdf_doc[page_index]
                        for img_index, img_info in enumerate(page.get_images(full=True)):
                            xref = img_info[0]
                            base_image = pdf_doc.extract_image(xref)
                            image_data = base_image["image"]
                            pil_img = Image.open(io.BytesIO(image_data))
                            if pil_img.width < 50 or pil_img.height < 50:
                                image_objects.append((f"Small Image {len(image_objects) + 1} (Page {page_index + 1})", pil_img))
                                continue
                            image_stream = io.BytesIO(image_data)
                            analysis_result = processor.vision_client.analyze_image_in_stream(image_stream, [VisualFeatureTypes.description])
                            image_name = get_image_name(analysis_result, len(image_objects) + 1)
                            image_objects.append((f"{image_name} (Page {page_index + 1})", pil_img))
                elif uploaded_file.type in ["image/png", "image/jpeg"]:
                    image = Image.open(io.BytesIO(file_bytes))
                    if image.width < 50 or image.height < 50:
                        image_objects.append((f"Small Image {len(image_objects) + 1}", image))
                    else:
                        image_bytes = io.BytesIO()
                        image.save(image_bytes, format=uploaded_file.type.split("/")[1])
                        image_bytes.seek(0)
                        analysis_result = processor.vision_client.analyze_image_in_stream(image_bytes, [VisualFeatureTypes.description])
                        image_name = get_image_name(analysis_result, len(image_objects) + 1)
                        image_objects.append((image_name, image))

                for name, img in image_objects:
                    st.write(f"**{name}**")
                    st.image(img, use_container_width=True)

            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
                st.stop()

if __name__ == "__main__":
    main()
