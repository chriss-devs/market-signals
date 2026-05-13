"""FastAPI routes for the Spanish RAG chatbot panel."""

from __future__ import annotations

from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

router = APIRouter(prefix="/chatbot")

ALLOWED_EXTENSIONS = {"pdf", "docx", "doc", "txt"}


def _ext(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


# ── Pages ──────────────────────────────────────────────────────────────────────

@router.get("", response_class=HTMLResponse)
async def chatbot_page(request: Request):
    return request.app.state.templates.TemplateResponse("chatbot.html", {
        "request": request,
        "stats": None,
    })


# ── Documents API ──────────────────────────────────────────────────────────────

@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    ext = _ext(file.filename or "")
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de archivo no permitido. Use: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    content = await file.read()
    if len(content) > 20 * 1024 * 1024:  # 20 MB limit
        raise HTTPException(status_code=400, detail="El archivo supera el límite de 20 MB.")

    from market_signal_engine.chatbot.document_processor import extract_text, chunk_text
    from market_signal_engine.chatbot.rag import embed_batch
    from market_signal_engine.chatbot.db import save_document, save_chunks
    from market_signal_engine.database.connection import get_session

    text = extract_text(file.filename, content)
    if not text.strip():
        raise HTTPException(status_code=400, detail="No se pudo extraer texto del archivo.")

    chunks = chunk_text(text)
    embeddings = embed_batch(chunks)

    session = get_session()
    try:
        doc_id = save_document(session, file.filename, ext, len(text))
        save_chunks(session, doc_id, chunks, embeddings)
    finally:
        session.close()

    return JSONResponse({
        "ok": True,
        "id": doc_id,
        "filename": file.filename,
        "chunks": len(chunks),
    })


@router.get("/documents")
async def list_documents():
    from market_signal_engine.chatbot.db import list_documents as _list
    from market_signal_engine.database.connection import get_session

    session = get_session()
    try:
        docs = _list(session)
    finally:
        session.close()

    return JSONResponse([
        {
            "id": d.id,
            "filename": d.filename,
            "file_type": d.file_type,
            "char_count": d.char_count,
            "created_at": d.created_at.isoformat() if d.created_at else "",
        }
        for d in docs
    ])


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: int):
    from market_signal_engine.chatbot.db import delete_document as _delete
    from market_signal_engine.database.connection import get_session

    session = get_session()
    try:
        deleted = _delete(session, doc_id)
    finally:
        session.close()

    if not deleted:
        raise HTTPException(status_code=404, detail="Documento no encontrado.")
    return JSONResponse({"ok": True})


# ── Chat API ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str


@router.post("/chat")
async def chat(body: ChatRequest):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")

    from market_signal_engine.chatbot.rag import answer
    from market_signal_engine.database.connection import get_session

    session = get_session()
    try:
        reply = answer(question, session)
    finally:
        session.close()

    return JSONResponse({"answer": reply})
