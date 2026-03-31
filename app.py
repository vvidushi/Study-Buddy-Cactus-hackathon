"""
app.py — FastAPI server for Study Buddy
==============================================================
Architecture:
  Study mode  → PDF text written to corpus dir → cactus_init(corpus_dir) embeds it
                cactus_rag_query retrieves top-k chunks → FunctionGemma answers from context
  Practice    → FunctionGemma tool-calling for structured score output
  STT         → Whisper-tiny via cactus_transcribe

Endpoints:
  GET  /            → mobile UI
  POST /upload-pdf  → extract PDF → write corpus.txt → init RAG model
  POST /ask         → cactus_rag_query + FunctionGemma answer
  POST /quiz        → pick a sentence from PDF as practice question
  POST /feedback    → FunctionGemma scores the student's answer
  POST /transcribe  → Whisper STT
  GET  /health
"""

import os
import re
import sys
import json
import tempfile
import shutil
from pathlib import Path

from dotenv import load_dotenv

# Load repo-root .env into the process (secrets never committed; see .env.example)
load_dotenv(Path(__file__).resolve().parent / ".env")

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from stt import SpeechToText
from coach import StudyBuddy
from pdf_reader import extract_text, get_relevant_context

# ── Cactus SDK ───────────────────────────────────────────────────────────────
CACTUS_REPO = Path(__file__).resolve().parent.parent / "cactus"
sys.path.insert(0, str(CACTUS_REPO / "python" / "src"))
from cactus import cactus_init, cactus_complete, cactus_rag_query, cactus_destroy, cactus_reset

LLM_PATH = str(CACTUS_REPO / "weights" / "functiongemma-270m-it")

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Study Buddy")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# ── Models ───────────────────────────────────────────────────────────────────
print("Loading on-device models…")
try:
    stt = SpeechToText()
    print("  ✓ Whisper (STT) ready")
except RuntimeError as e:
    stt = None
    print(f"  ✗ Whisper not available: {e}")

study_buddy = StudyBuddy()
print("  ✓ FunctionGemma (scoring) ready")

# ── RAG state ─────────────────────────────────────────────────────────────────
_pdf_text    = ""
_pdf_name    = ""
_rag_model   = None          # FunctionGemma instance initialised with corpus_dir
_corpus_dir  = None          # temp dir holding corpus.txt for RAG

TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"
print("\nStudy Buddy is running at http://localhost:8000\n")


# ── Exception handler for upload validation ───────────────────────────────────
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    if request.url.path == "/upload-pdf" and exc.errors():
        msg = "Please select a PDF file and try again. (Form field must be 'pdf'.)"
        return JSONResponse(status_code=422, content={"detail": msg})
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    if not TEMPLATE_PATH.exists():
        return HTMLResponse("<h1>UI not found</h1>")
    return HTMLResponse(content=TEMPLATE_PATH.read_text())


@app.get("/health")
async def health():
    return {
        "status":     "ok",
        "stt_ready":  stt is not None,
        "pdf_loaded": bool(_pdf_text),
        "pdf_name":   _pdf_name or None,
        "rag_ready":  _rag_model is not None,
        "on_device":  True,
    }


@app.post("/upload-pdf")
async def upload_pdf(pdf: UploadFile = File(...)):
    """Extract PDF text, write corpus, and initialise the RAG model."""
    global _pdf_text, _pdf_name, _rag_model, _corpus_dir

    if not pdf.filename or not str(pdf.filename).lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await pdf.read()
    if not content:
        raise HTTPException(status_code=400, detail="File is empty. Please choose a valid PDF.")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_pdf = tmp.name

    try:
        text = extract_text(tmp_pdf)
        if not text.strip():
            raise HTTPException(status_code=400, detail="PDF appears to be empty or image-only.")

        # ── Tear down old RAG model if one exists ──────────────────────────
        if _rag_model is not None:
            try:
                cactus_destroy(_rag_model)
            except Exception:
                pass
            _rag_model = None

        # ── Write corpus file ──────────────────────────────────────────────
        if _corpus_dir is None:
            _corpus_dir = tempfile.mkdtemp(prefix="studybuddy_corpus_")
        corpus_file = Path(_corpus_dir) / "corpus.txt"
        corpus_file.write_text(text, encoding="utf-8")

        # ── Init FunctionGemma with corpus_dir for RAG ─────────────────────
        print(f"  [RAG] Indexing corpus ({len(text)} chars)…")
        _rag_model = cactus_init(LLM_PATH, corpus_dir=_corpus_dir, cache_index=False)
        if _rag_model is None:
            print("  [RAG] Warning: RAG model init failed, will fall back to keyword search.")

        _pdf_text = text
        _pdf_name = pdf.filename
        print(f"  [RAG] Ready — {pdf.filename}")
        return JSONResponse({"success": True, "filename": pdf.filename, "chars": len(text)})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_pdf)
        except OSError:
            pass


class AskRequest(BaseModel):
    question: str


def _is_greeting(question: str) -> bool:
    """Return True if the question has no meaningful study-related keywords."""
    stop = {
        "a","an","the","is","are","was","were","be","been","being","have","has","had",
        "do","does","did","will","would","could","should","may","might","shall",
        "what","when","where","who","which","how","why","and","or","but","in","on",
        "at","to","for","of","with","by","from","about","this","that","these","those",
        "my","your","his","her","its","our","their","i","you","he","she","it","we",
        "they","me","him","us","them","not","no","s","t","don","isn","can","just",
        "also","very","more","than","hey","hi","hello","sup","yes","no","ok","okay"
    }
    words = set(re.findall(r'\b\w+\b', question.lower()))
    meaningful = {w for w in words - stop if len(w) > 2}
    greetings  = {"hey","hi","hello","sup","whats","there","guys","bro"}
    return not meaningful or meaningful <= greetings


@app.post("/ask")
async def ask(req: AskRequest):
    """
    Study mode: RAG retrieval via cactus_rag_query → FunctionGemma answer.
    Falls back to keyword search if RAG model is not available.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Please ask a question.")

    if not _pdf_text:
        return JSONResponse({
            "success":    True,
            "answer":     "Please upload your study PDF first using the button at the top.",
            "key_points": [],
        })

    if _is_greeting(req.question):
        return JSONResponse({
            "success":    True,
            "answer":     'Ask me something about your PDF — e.g. "What is Big Data?", "Explain chapter 1", "What are the key topics?"',
            "key_points": [],
        })

    # ── Step 1: RAG retrieval ─────────────────────────────────────────────────
    context = ""
    if _rag_model is not None:
        try:
            chunks = cactus_rag_query(_rag_model, req.question, top_k=4)
            if chunks:
                context = "\n\n".join(c["text"] for c in chunks if c.get("text"))[:1200]
                print(f"  [RAG] Retrieved {len(chunks)} chunks")
        except Exception as e:
            print(f"  [RAG] Query error: {e}")

    # Fallback: keyword search if RAG returned nothing
    if not context:
        context = get_relevant_context(_pdf_text, req.question, max_chars=800)

    # ── Step 2: FunctionGemma generates an answer from context ────────────────
    answer = ""
    if _rag_model is not None:
        try:
            prompt = (
                f"You are a study assistant. Answer the question using ONLY the study material below. "
                f"Be concise and clear.\n\n"
                f"Study material:\n{context}\n\n"
                f"Question: {req.question}\n\nAnswer:"
            )
            raw = cactus_complete(
                _rag_model,
                [{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.2,
                stop_sequences=["<|im_end|>", "<end_of_turn>", "\n\nQuestion:"],
            )
            result = json.loads(raw)
            answer = (result.get("response") or "").strip()
            print(f"  [LLM] Answer: {answer[:80]}…")
        except Exception as e:
            print(f"  [LLM] Generation error: {e}")

    # Fallback: return context directly if LLM gave nothing
    if not answer:
        sentences = re.split(r'(?<=[.!?])\s+', context)
        sentences = [s.strip() for s in sentences if s.strip()]
        main_parts, total = [], 0
        for s in sentences:
            if total + len(s) <= 600:
                main_parts.append(s)
                total += len(s)
            else:
                break
        answer = " ".join(main_parts) if main_parts else context[:600].strip()

    # ── Step 3: Extract key points from the retrieved context ─────────────────
    ctx_sentences = re.split(r'(?<=[.!?])\s+', context)
    bullets = [s.strip() for s in ctx_sentences if len(s.strip()) > 50][:3]

    return JSONResponse({
        "success":    True,
        "answer":     answer,
        "key_points": bullets,
    })


class QuizRequest(BaseModel):
    topic: str = ""


@app.post("/quiz")
async def quiz(req: QuizRequest):
    """Practice mode: pick a sentence from the PDF as a practice question."""
    if _pdf_text:
        query   = req.topic.strip() or "definition concept example"
        passage = get_relevant_context(_pdf_text, query, max_chars=600)
        sentences = re.split(r'(?<=[.!?])\s+', passage)
        long_s = [s.strip() for s in sentences if len(s.strip()) > 50]
        if long_s:
            return JSONResponse({
                "success":  True,
                "question": f"Explain in your own words: \"{long_s[0][:150]}\"",
                "hint":     long_s[1][:100] if len(long_s) > 1 else "Think about the key terms.",
            })

    return JSONResponse({
        "success":  True,
        "question": "Explain the main topic covered in your study material.",
        "hint":     "Use specific terms and examples from the text.",
    })


class FeedbackRequest(BaseModel):
    transcript: str
    question:   str = ""


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    """Practice mode: FunctionGemma scores the student's answer via tool-calling."""
    pdf_context = ""
    if _pdf_text and (req.question or req.transcript):
        query = req.question or req.transcript
        pdf_context = get_relevant_context(_pdf_text, query, max_chars=400)

    try:
        result = study_buddy.get_feedback(
            transcript=req.transcript,
            question=req.question,
            pdf_context=pdf_context,
        )
        return JSONResponse({
            "score":    result["score"],
            "bullets":  result["bullets"],
            "used_pdf": bool(pdf_context),
            "success":  True,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    if stt is None:
        raise HTTPException(status_code=503, detail="Whisper model not loaded.")

    suffix = ".wav"
    ct = audio.content_type or ""
    if "mp4" in ct:    suffix = ".mp4"
    elif "webm" in ct: suffix = ".webm"

    audio_bytes = await audio.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    print(f"[STT] Received audio: {len(audio_bytes)} bytes, suffix={suffix}")

    try:
        transcript = stt.transcribe(tmp_path)
        if not transcript:
            return JSONResponse({
                "transcript": "",
                "success":    False,
                "detail":     "No speech detected. Please speak clearly and try again.",
            }, status_code=200)
        return JSONResponse({"transcript": transcript, "success": True})
    except Exception as e:
        print(f"  [STT] transcribe endpoint error: {e}")
        return JSONResponse({
            "transcript": "",
            "success":    False,
            "detail":     "Speech recognition failed. Please try again or type your question.",
        }, status_code=200)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
