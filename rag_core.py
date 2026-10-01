"""Reusable backend for the Document Q&A Streamlit application."""

from __future__ import annotations

import io
import re
import time
from collections import Counter
from dataclasses import dataclass
from statistics import mean
from typing import Callable

import faiss
import numpy as np
import pymupdf as fitz
import pytesseract
import requests
from google import genai
from google.genai import errors, types
from PIL import Image, ImageFilter, ImageOps
from pytesseract import Output


OCR_SPACE_ENDPOINT = "https://api.ocr.space/parse/image"
OCR_SPACE_MAX_BYTES = 1_000_000
MIN_NATIVE_TEXT_CHARS = 120
OCR_SCALE = 3
OCR_CONFIG = "--oem 3 --psm 6"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
DEFAULT_MODELS = (
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
)

ProgressCallback = Callable[[int, int, str], None]


@dataclass
class ProcessedDocument:
    filename: str
    page_count: int
    pages: list[dict]
    chunks: list[dict]
    embeddings: np.ndarray
    index: faiss.Index
    extraction_methods: dict[str, int]
    warnings: list[str]


def _notify(callback: ProgressCallback | None, done: int, total: int, text: str):
    if callback:
        callback(done, total, text)


def _render_for_managed_ocr(page) -> bytes:
    for scale, quality in [(2.0, 88), (2.0, 75), (1.7, 75), (1.5, 65), (1.3, 55)]:
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        data = buffer.getvalue()
        if len(data) <= OCR_SPACE_MAX_BYTES:
            return data
    raise ValueError("The page image could not be compressed below OCR.Space's 1 MB limit.")


def _ocr_space_page(page, page_number: int, api_key: str) -> str:
    response = requests.post(
        OCR_SPACE_ENDPOINT,
        headers={"apikey": api_key},
        files={
            "file": (
                f"page_{page_number}.jpg",
                _render_for_managed_ocr(page),
                "image/jpeg",
            )
        },
        data={
            "OCREngine": "3",
            "language": "auto",
            "isOverlayRequired": "false",
            "detectOrientation": "true",
            "scale": "true",
        },
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("IsErroredOnProcessing"):
        detail = payload.get("ErrorMessage") or payload.get("ErrorDetails")
        raise RuntimeError(f"OCR.Space rejected page {page_number}: {detail}")
    return "\n".join(
        (item.get("ParsedText") or "").strip()
        for item in (payload.get("ParsedResults") or [])
        if (item.get("ParsedText") or "").strip()
    ).strip()


def _tesseract_page(page) -> tuple[str, float]:
    pix = page.get_pixmap(matrix=fitz.Matrix(OCR_SCALE, OCR_SCALE), alpha=False)
    image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
    image = ImageOps.autocontrast(image).filter(ImageFilter.SHARPEN)
    text = pytesseract.image_to_string(image, config=OCR_CONFIG).strip()
    data = pytesseract.image_to_data(image, config=OCR_CONFIG, output_type=Output.DICT)
    confidences = []
    for raw_confidence, token in zip(data["conf"], data["text"]):
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            continue
        if token.strip() and confidence >= 0:
            confidences.append(confidence)
    return text, mean(confidences) if confidences else 0.0


def extract_pages(
    pdf_bytes: bytes,
    ocr_space_api_key: str = "",
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[dict], int, list[str]]:
    """Extract native text first and OCR only pages that need it."""
    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"The uploaded file is not a readable PDF: {exc}") from exc

    if document.needs_pass:
        document.close()
        raise ValueError("Password-protected PDFs are not supported in this demo.")

    page_count = len(document)
    if not page_count:
        document.close()
        raise ValueError("The PDF contains no pages.")

    pages: list[dict] = []
    warnings: list[str] = []

    try:
        for page_number, page in enumerate(document, start=1):
            _notify(
                progress_callback,
                page_number - 1,
                page_count,
                f"Extracting page {page_number} of {page_count}",
            )
            native_text = page.get_text("text").strip()
            text = native_text
            method = "native"
            confidence = None

            if len(native_text) < MIN_NATIVE_TEXT_CHARS:
                if ocr_space_api_key:
                    try:
                        text = _ocr_space_page(page, page_number, ocr_space_api_key)
                        method = "ocr_space_engine_3"
                    except Exception as exc:
                        warnings.append(f"Page {page_number}: managed OCR failed ({exc}).")
                        try:
                            text, confidence = _tesseract_page(page)
                            method = "tesseract_fallback"
                        except Exception as fallback_exc:
                            warnings.append(
                                f"Page {page_number}: Tesseract fallback failed ({fallback_exc})."
                            )
                            text = native_text
                            method = "native"
                else:
                    try:
                        text, confidence = _tesseract_page(page)
                        method = "tesseract_fallback"
                    except Exception as exc:
                        warnings.append(f"Page {page_number}: OCR unavailable ({exc}).")
                        text = native_text
                        method = "native"

                if len(text.strip()) < len(native_text):
                    text = native_text
                    method = "native"
                    confidence = None

            text = text.strip()
            if text:
                pages.append(
                    {
                        "page": page_number,
                        "text": text,
                        "extraction_method": method,
                        "ocr_confidence": confidence,
                        "char_count": len(text),
                    }
                )
            else:
                warnings.append(f"Page {page_number}: no text was extracted.")
    finally:
        document.close()

    if not pages:
        raise ValueError("No readable text could be extracted from this PDF.")

    _notify(progress_callback, page_count, page_count, "Text extraction complete")
    return pages, page_count, warnings


def _clean_text(text: str) -> str:
    text = text.replace("\x0c", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _choose_chunk_end(text: str, start: int, proposed_end: int) -> int:
    if proposed_end >= len(text):
        return len(text)
    minimum = start + int(CHUNK_SIZE * 0.60)
    candidates = [
        text.rfind("\n\n", minimum, proposed_end),
        text.rfind("\n", minimum, proposed_end),
        text.rfind(". ", minimum, proposed_end),
        text.rfind("? ", minimum, proposed_end),
        text.rfind("! ", minimum, proposed_end),
    ]
    boundary = max(candidates)
    return boundary + 1 if boundary >= minimum else proposed_end


def create_chunks(pages: list[dict]) -> list[dict]:
    chunks: list[dict] = []
    for page in pages:
        text = _clean_text(page["text"])
        start = 0
        while start < len(text):
            proposed_end = min(start + CHUNK_SIZE, len(text))
            end = _choose_chunk_end(text, start, proposed_end)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    {
                        "chunk_id": len(chunks),
                        "page": page["page"],
                        "text": chunk_text,
                        "start_char": start,
                        "end_char": end,
                    }
                )
            if end >= len(text):
                break
            start = max(end - CHUNK_OVERLAP, start + 1)
    if not chunks:
        raise ValueError("No searchable chunks could be created from the extracted text.")
    return chunks


def process_document(
    pdf_bytes: bytes,
    filename: str,
    embedding_model,
    ocr_space_api_key: str = "",
    progress_callback: ProgressCallback | None = None,
) -> ProcessedDocument:
    pages, page_count, warnings = extract_pages(
        pdf_bytes,
        ocr_space_api_key=ocr_space_api_key,
        progress_callback=progress_callback,
    )
    _notify(progress_callback, 1, 3, "Creating page-aware chunks")
    chunks = create_chunks(pages)
    _notify(progress_callback, 2, 3, "Creating semantic embeddings")
    embeddings = embedding_model.encode(
        [chunk["text"] for chunk in chunks],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    _notify(progress_callback, 3, 3, "Document index ready")
    return ProcessedDocument(
        filename=filename,
        page_count=page_count,
        pages=pages,
        chunks=chunks,
        embeddings=embeddings,
        index=index,
        extraction_methods=dict(Counter(page["extraction_method"] for page in pages)),
        warnings=warnings,
    )


def retrieve_chunks(
    question: str,
    processed: ProcessedDocument,
    embedding_model,
    top_k: int = 5,
) -> list[dict]:
    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
    k = min(top_k, len(processed.chunks))
    scores, indices = processed.index.search(question_embedding, k)
    results = []
    for score, chunk_index in zip(scores[0], indices[0]):
        if chunk_index < 0:
            continue
        item = dict(processed.chunks[int(chunk_index)])
        item["score"] = float(score)
        results.append(item)
    return results


def _build_prompt(question: str, evidence: list[dict]) -> str:
    blocks = []
    for rank, item in enumerate(evidence, start=1):
        blocks.append(
            f"[SOURCE {rank} | PAGE {item['page']} | "
            f"CHUNK {item['chunk_id']} | SCORE {item['score']:.4f}]\n{item['text']}"
        )
    context = "\n\n".join(blocks)
    return f"""You are a document question-answering assistant.

Answer the question using ONLY the SOURCE excerpts below.

Rules:
- Do not use outside knowledge.
- If the evidence is insufficient, respond exactly: I cannot answer this from the provided document.
- Cite every factual statement using an original page number such as [Page 2].
- Never cite a page not present in the SOURCE excerpts.
- Keep the complete response under 180 words.
- Finish with a section named Supporting excerpts containing 1-2 short verbatim excerpts and page numbers.
- OCR text can contain mistakes. Never invent missing words.

QUESTION:
{question}

SOURCE EXCERPTS:
{context}
"""


def _finish_reason(response) -> str:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return "UNKNOWN"
    reason = getattr(candidates[0], "finish_reason", None)
    return str(getattr(reason, "value", reason) or "UNKNOWN")


def _validate_citations(answer: str, evidence: list[dict]) -> list[int]:
    allowed = {item["page"] for item in evidence}
    cited = {int(value) for value in re.findall(r"\[Page\s+(\d+)\]", answer)}
    invalid = cited - allowed
    if invalid:
        raise ValueError(f"The answer cited pages outside the retrieved evidence: {sorted(invalid)}")
    return sorted(cited)


def _generate_with_fallback(client, prompt: str, model_candidates: tuple[str, ...]):
    failures = []
    for model_name in model_candidates:
        for attempt in range(1, 3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=1600,
                    ),
                )
                return response, model_name
            except errors.ServerError as exc:
                failures.append(f"{model_name} attempt {attempt}: {exc}")
                if attempt < 2:
                    time.sleep(2)
            except errors.ClientError as exc:
                message = str(exc)
                if any(code in message for code in ("429", "404", "RESOURCE_EXHAUSTED")):
                    failures.append(f"{model_name}: {exc}")
                    break
                raise
    summary = "\n".join(failures[-6:])
    raise RuntimeError("All Gemini model attempts failed. Try again shortly.\n" + summary)


def answer_question(
    question: str,
    processed: ProcessedDocument,
    embedding_model,
    gemini_api_key: str,
    requested_model: str = "",
    top_k: int = 5,
) -> dict:
    question = question.strip()
    if not question:
        raise ValueError("Enter a question first.")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not configured.")

    evidence = retrieve_chunks(question, processed, embedding_model, top_k=top_k)
    prompt = _build_prompt(question, evidence)
    client = genai.Client(api_key=gemini_api_key)
    model_candidates = (requested_model,) if requested_model else DEFAULT_MODELS
    last_problem = "No answer was generated."

    for _ in range(2):
        response, model_used = _generate_with_fallback(client, prompt, model_candidates)
        answer = (response.text or "").strip()
        cited_pages = _validate_citations(answer, evidence)
        finish_reason = _finish_reason(response)
        refused = answer == "I cannot answer this from the provided document."
        complete = refused or (
            bool(answer)
            and bool(cited_pages)
            and "supporting excerpts" in answer.lower()
            and "MAX_TOKENS" not in finish_reason.upper()
        )
        if complete:
            return {
                "question": question,
                "answer": answer,
                "cited_pages": cited_pages,
                "sources": evidence,
                "model": model_used,
            }
        last_problem = f"Incomplete model response ({finish_reason})."
        prompt += (
            "\nThe previous response was incomplete. Answer in under 140 words, "
            "include citations, and finish the Supporting excerpts section."
        )
    raise RuntimeError(last_problem + " Please retry the question.")
