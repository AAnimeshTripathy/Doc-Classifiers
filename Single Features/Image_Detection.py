import streamlit as st
import requests
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from dotenv import load_dotenv
import os
from PIL import Image
import io
import fitz  # PyMuPDF for PDF image extraction
from pdf2image import convert_from_bytes

# Load environment variables from .env file
load_dotenv()

# Azure Configuration from .env file
AZURE_FORM_RECOGNIZER_ENDPOINT = os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT")
AZURE_FORM_RECOGNIZER_KEY = os.getenv("AZURE_FORM_RECOGNIZER_KEY")
AZURE_COMPUTER_VISION_ENDPOINT = os.getenv("AZURE_COMPUTER_VISION_ENDPOINT")
AZURE_COMPUTER_VISION_KEY = os.getenv("AZURE_COMPUTER_VISION_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")

# Streamlit App Title
st.title("AI-Powered Document Analysis System")

# File Upload Widget
uploaded_file = st.file_uploader("Upload your document (PDF, Image)", type=["pdf", "png", "jpg"])

# Function to extract image name from the analysis result, enhancing generic captions
def get_image_name(analysis_result, index):
    if analysis_result.description and analysis_result.description.captions:
        caption = analysis_result.description.captions[0].text.strip()
        # Define a list of generic terms that might be too vague
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

if uploaded_file is not None:
    st.info("Processing document...")

    # Read the file into memory once
    file_bytes = uploaded_file.read()
    if not file_bytes:
        st.error("Uploaded file is empty. Please upload a valid document.")
        st.stop()

    # -------------------------------------------------------------------------
    # Step 1: Extract Text & Tables Using Azure Form Recognizer (OCR)
    # -------------------------------------------------------------------------
    document_analysis_client = DocumentAnalysisClient(
        endpoint=AZURE_FORM_RECOGNIZER_ENDPOINT,
        credential=AzureKeyCredential(AZURE_FORM_RECOGNIZER_KEY)
    )
    
    # Pass the file bytes to the recognizer
    poller = document_analysis_client.begin_analyze_document(
        model_id="prebuilt-layout", document=file_bytes
    )
    result = poller.result()

    if not result:
        st.error("Failed to process document. Please try again.")
        st.stop()

    # Extract text from the OCR result
    extracted_text = "\n".join([line.content for page in result.pages for line in page.lines])
    # st.subheader("Extracted Text:")
    # st.write(extracted_text)

    # Identify Tables with AI-generated Names (using OpenAI as before)
    tables = []
    for i, table in enumerate(result.tables):
        table_data = []
        for cell in table.cells:
            row, col = cell.row_index, cell.column_index
            while len(table_data) <= row:
                table_data.append([])
            while len(table_data[row]) <= col:
                table_data[row].append("")
            table_data[row][col] = cell.content

        headers_text = " ".join(table_data[0]) if table_data else "Table"
        prompt = f"Analyze the following headers and suggest a meaningful table name: {headers_text}"
        headers_req = {
            "Content-Type": "application/json",
            "api-key": AZURE_OPENAI_KEY
        }
        data = {
            "messages": [
                {"role": "system", "content": "You are an AI assistant helping to classify tables."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 50,
            "temperature": 0.7
        }
        response = requests.post(AZURE_OPENAI_ENDPOINT, headers=headers_req, json=data)
        if response.status_code == 200:
            table_name = response.json().get("choices")[0].get("message").get("content", f"Table {i+1}")
        else:
            table_name = f"Table {i+1}"
        tables.append((table_name, table_data))

    st.subheader("Identified Tables:")
    for table_name, table_data in tables:
        st.write(f"**{table_name}**")
        st.table(table_data)

    # -------------------------------------------------------------------------
    # Step 2: Extract and Analyze Images
    # -------------------------------------------------------------------------
    vision_client = ComputerVisionClient(
        AZURE_COMPUTER_VISION_ENDPOINT,
        CognitiveServicesCredentials(AZURE_COMPUTER_VISION_KEY)
    )
    visual_features = [VisualFeatureTypes.objects, VisualFeatureTypes.tags, VisualFeatureTypes.description]
    image_objects = []

    # =========================
    # A) If the file is a PDF
    # =========================
    if uploaded_file.type == "application/pdf":
        pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
        page_count = pdf_doc.page_count

        for page_index in range(page_count):
            page = pdf_doc[page_index]
            image_list = page.get_images(full=True)
            if not image_list:
                continue

            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = pdf_doc.extract_image(xref)
                image_data = base_image["image"]

                # Open the image using PIL to check its dimensions
                pil_img = Image.open(io.BytesIO(image_data))
                if pil_img.width < 50 or pil_img.height < 50:
                    st.warning(f"Image on Page {page_index + 1} is too small for analysis. Skipping analysis for this image.")
                    image_objects.append((f"Small Image {len(image_objects) + 1} (Page {page_index + 1})", pil_img))
                    continue

                image_stream = io.BytesIO(image_data)
                analysis_result = vision_client.analyze_image_in_stream(image_stream, visual_features)
                image_name = get_image_name(analysis_result, len(image_objects) + 1)
                image_objects.append((f"{image_name} (Page {page_index + 1})", pil_img))

    # =========================
    # B) If the file is an Image
    # =========================
    elif uploaded_file.type in ["image/png", "image/jpeg"]:
        image = Image.open(io.BytesIO(file_bytes))
        if image.width < 50 or image.height < 50:
            st.warning("Uploaded image is too small for analysis. Skipping analysis.")
            image_objects.append((f"Small Image {len(image_objects) + 1}", image))
        else:
            image_bytes = io.BytesIO()
            image.save(image_bytes, format=uploaded_file.type.split("/")[1])
            image_bytes.seek(0)
            analysis_result = vision_client.analyze_image_in_stream(image_bytes, visual_features)
            image_name = get_image_name(analysis_result, len(image_objects) + 1)
            image_objects.append((image_name, image))

    st.subheader("Identified Images, Graphs & Charts:")
    if image_objects:
        for name, img in image_objects:
            st.write(f"**{name}**")
            st.image(img, use_container_width=True)
    else:
        st.info("No embedded images found in this PDF (or no objects recognized).")

    st.success("Document analysis completed!")
