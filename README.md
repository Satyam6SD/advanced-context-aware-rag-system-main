# Advanced Rag thorugh api

This project implements a Retrieval-Augmented Generation (RAG). It utilizes a Python backend with Flask to provide an API endpoint that processes user questions and a document URL to deliver context-aware, enhanced answers.

***

## 🚀 Features

* **RAG Pipeline**: Implements an end-to-end RAG pipeline using LangChain.
* **Vector Database**: Uses Pinecone as a serverless vector database for efficient document retrieval.
* **Language Model**: Leverages the GROQ API to generate high-quality responses.
* **Hybrid Search**: Integrates Pinecone's hybrid search for a combination of dense and sparse retrieval to improve search accuracy.
* **PDF/DOCX Processing**: Supports document processing from remote URLs using `PyMuPDFLoader` and `Docx2txtLoader`.
* **RESTful API**: Provides a clean and simple API endpoint for integration with other services.

***

## ⚙️ Prerequisites

Before running this project, ensure you have the following installed:

* Python 3.9+
* `pip` (Python package installer)

You will also need API keys for the following services:

* GROQ API Key
* Pinecone API Key

***

## 💻 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/23shivay/advanced-context-aware-rag-system
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    The project relies on a number of libraries. Although a `requirements.txt` file was not provided, based on the code, the following dependencies are required:
    ```bash
    pip install Flask flask_cors python-dotenv langchain_groq langchain-huggingface pinecone-client pinecone_text pydantic pymupdf docx2txt
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the root directory of your project and add your API keys.
    ```
    GROQ_API_KEY="your_groq_api_key"
    PINECONE_API_KEY="your_pinecone_api_key"
    PINECONE_ENVIRONMENT="us-east-1" # Or your specific Pinecone environment
    ```

***

## ▶️ Usage

### Running the API Server

The API is built using Flask. To start the server, you can run the `rag_api.py` script.

```bash
python rag_api.py
