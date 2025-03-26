# Doc-Classifiers

## Team Information

**Team Name:** Doc-Classifiers

**Members:**
- Kashika Agarwal (Leader)
- Animesh Tripathy
- Amitesh Raj
- Siddharth Srivastava
- Faizal

## Project Deployed On

The project is deployed and accessible at: [Doc-Classifiers App](https://doc-classifiers.streamlit.app/)

## Demo

A video demonstration of the project is available at: [Demo Video](https://drive.google.com/file/d/1zirau03TG5Uw-IfOgVOAlqQw7uKCMY9V/view?usp=sharing)

For Research Documentation , refer to [Working Document](https://docs.google.com/document/d/1Dl_bjsQq41KzWfLyvFCu8JYFdLnUXUdSsmfRmwboSNc/edit?usp=sharing).


## Project Overview

Doc-Classifiers is an AI-powered document processing system that utilizes Azure AI services to analyze, classify, and summarize documents. The system can handle various document types including PDF, DOCX, PNG, and JPG files.

## Features

1. **Text Extraction**: Extracts text from uploaded documents using Azure Form Recognizer.
2. **Document Summarization**: Generates overall and section-wise summaries using Azure OpenAI.
3. **Document Classification**: Categorizes documents and provides relevant tags.
4. **Table Analysis**: Detects and summarizes tables within documents.
5. **Keyword Extraction**: Identifies and defines key terms from the document.
6. **Citation and Reference Extraction**: Extracts citations, references, and links from the document.
7. **Image Analysis**: Processes images within documents, providing descriptions and object detection.

## Technologies Used

- Azure Form Recognizer (Document Intelligence)
- Azure OpenAI
- Azure Text Analytics
- Azure Computer Vision
- Streamlit
- Python

## How to Run

1. Create a virtual environment:
    ```sh
    python -m venv venv
    ```

2. Activate the virtual environment:
    - On Windows:
        ```sh
        venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```sh
        source venv/bin/activate
        ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Set up environment variables:
   Create a `.env` file in the project root and add the following variables:
   ```
   AZURE_FORM_RECOGNIZER_ENDPOINT=
   AZURE_FORM_RECOGNIZER_KEY=
   AZURE_OPENAI_ENDPOINT=
   AZURE_OPENAI_KEY=
   AZURE_OPENAI_DEPLOYMENT=
   AZURE_LANGUAGE_ENDPOINT=
   AZURE_LANGUAGE_KEY=
   AZURE_COMPUTER_VISION_ENDPOINT=
   AZURE_COMPUTER_VISION_KEY=
   ```

5. Run the Streamlit application:
    ```sh
    streamlit run project.py
    ```

## Usage

1. Upload a document (PDF, DOCX, PNG, or JPG) using the file uploader.
2. The system will process the document and display the following information:
   - Overall Summary
   - Document Classification
   - Section-wise Summary
   - Table Analysis (if tables are present)
   - Key Terms
   - References & Links
   - Images (if present in the document)
