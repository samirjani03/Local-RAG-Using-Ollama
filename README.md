# Local PDF RAG Assistant 📚

A fully local, privacy-first Retrieval-Augmented Generation (RAG) application that allows you to upload large PDF documents (up to 300+ pages) and chat with them. It operates entirely on your machine, ensuring 100% data privacy with zero API costs.

This project is built using:
- **Streamlit** for the frontend UI.
- **LangChain** for RAG orchestration.
- **ChromaDB** for the local Vector Database.
- **Ollama** for Local LLMs (`llama3.1:8b`) and Embeddings (`nomic-embed-text`).

---

## 🧠 How the Backend Works

When you interact with this app, several steps happen under the hood:

1. **PDF Parsing (`PyPDFLoader`)**: When you upload a PDF, the app extracts the raw text from the document.
2. **Chunking (`RecursiveCharacterTextSplitter`)**: AI models have a "context window" (a limit to how much text they can process at once). To bypass this, the app splits your massive PDF into smaller, overlapping chunks (1000 characters each, with a 200-character overlap to preserve context).
3. **Embeddings (`OllamaEmbeddings - nomic-embed-text`)**: The app converts these text chunks into high-dimensional numerical vectors (lists of numbers). These numbers represent the semantic meaning of the text.
4. **Vector Database (`ChromaDB`)**: These vectors are saved into `ChromaDB` (inside the `chroma_db` folder). This acts as a search engine. Because it's "persistent", you only have to process your PDF once.
5. **Retrieval**: When you ask a question in the chat, your question is also turned into a vector. ChromaDB calculates the mathematical distance between your question's vector and the document chunks' vectors to find the most relevant pieces of text.
6. **Generation (`OllamaLLM - llama3.1:8b`)**: The app bundles your question and the retrieved chunks together and sends them to the local `llama3.1` model. The model reads the chunks and types out an accurate, context-aware answer!

---

## 🚀Step-by-Step Setup Guide

Follow these exact steps to get the app running on your own computer.

### Step 1: Install Python
Ensure you have Python installed on your system (Python 3.11 is recommended). You can download it from [python.org](https://www.python.org/downloads/).

### Step 2: Install Ollama and Pull Models
Ollama is the engine that runs the AI models locally.
1. Download and install Ollama from [ollama.com](https://ollama.com/).
2. Open your terminal (or command prompt) and run the following commands to download the necessary models. *(Note: These files are large, so it might take a few minutes depending on your internet speed).*
   ```bash
   ollama pull llama3.1:8b
   ollama pull nomic-embed-text
   ```
3. Make sure Ollama is running in the background before proceeding.

### Step 3: Set Up the Project
1. Clone this repository or download the project files.
2. Open your terminal and navigate to the project folder.
3. Create a virtual environment to keep your dependencies isolated:
   ```bash
   python -m venv .venv
   ```
4. Activate the virtual environment:
   - **Windows:**
     ```bash
     .\.venv\Scripts\activate
     ```
   - **Mac/Linux:**
     ```bash
     source .venv/bin/activate
     ```

### Step 4: Install Dependencies
With your virtual environment activated, install all the required Python libraries:
```bash
pip install -r requirements.txt
```

### Step 5: Run the App
Start the Streamlit application by running:
```bash
streamlit run app.py
```
A browser window will automatically open at `http://localhost:8501`. 

### Usage
1. Upload a PDF using the sidebar.
2. Click **"Process Document"** and wait for the success message.
3. Start asking questions in the chat!

---

## ⚠️ Notes for GitHub Contributors
- The `.venv` (virtual environment) and `chroma_db` (vector database) folders are purposefully ignored in `.gitignore`. **Do not upload them.** 
- Users cloning this repo must generate their own `chroma_db` by uploading their own PDFs.
