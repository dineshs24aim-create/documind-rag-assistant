# Document Q&A Assistant

A hackathon-ready Retrieval-Augmented Generation application that accepts arbitrary PDFs and returns grounded answers with page citations and supporting excerpts.

## Architecture

1. Native PDF text extraction with PyMuPDF.
2. OCR.Space Engine 3 for low-text/image pages, with local Tesseract fallback.
3. Page-aware 800-character chunks with 150-character overlap.
4. Normalized `all-MiniLM-L6-v2` embeddings.
5. FAISS inner-product search, equivalent to cosine similarity for normalized vectors.
6. Gemini answer generation restricted to the retrieved excerpts.
7. Citation validation before an answer is displayed.

## Run locally

From Terminal:

```bash
cd /Users/dineshsivasubramanian/Downloads/document_qa_app
python -m pip install -r requirements.txt
export OCR_SPACE_API_KEY="YOUR_PRIVATE_OCR_SPACE_KEY"
export GEMINI_API_KEY="YOUR_PRIVATE_GEMINI_KEY"
streamlit run app.py
```

The browser should open at `http://localhost:8501`. The included Streamlit
configuration disables the development file watcher because it can incorrectly
inspect optional vision modules inside `transformers` and report missing
`torchvision` even though this application is text-only.

The interface uses a high-contrast black and violet design optimized for a
projector-based hackathon demo.

Alternatively, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and replace the placeholder values. Never share or commit the real secrets file.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and push this folder to it.
2. Open [share.streamlit.io](https://share.streamlit.io) and select **Create app**.
3. Choose the repository, the `main` branch, and `app.py` as the entrypoint.
4. Under **Advanced settings**, select Python 3.12.
5. Add the following values under **Secrets**:

```toml
GEMINI_API_KEY = "your-real-gemini-key"
OCR_SPACE_API_KEY = "your-real-ocr-space-key"
```

`OCR_SPACE_API_KEY` is optional because `packages.txt` installs Tesseract as a
local fallback. Configured server-side secrets are never displayed in browser
fields.

## Demo flow

1. Upload a 10+ page PDF.
2. Select **Process document**.
3. Show the page, chunk, and extraction-method statistics.
4. Ask at least five questions.
5. Open **View retrieved evidence** to demonstrate page-level grounding.

## Known limitations

- OCR quality depends on scan quality and handwriting clarity.
- Tables, formulas, and complex multi-column layouts may require specialized parsing.
- OCR.Space and Gemini free tiers can temporarily rate-limit requests.
- The embedding model is optimized for English documents.
