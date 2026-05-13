"""Database helpers for RAG documents and chunks."""

from __future__ import annotations

import json
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.orm import Session

from market_signal_engine.database.connection import get_session

# ---------- ORM-free table definitions (avoid coupling to existing Base) --------

metadata = sa.MetaData()

rag_documents = sa.Table(
    "rag_documents",
    metadata,
    sa.Column("id", sa.Integer(), primary_key=True),
    sa.Column("filename", sa.Text(), nullable=False),
    sa.Column("file_type", sa.Text(), nullable=False),
    sa.Column("char_count", sa.Integer()),
    sa.Column("created_at", sa.DateTime(), default=datetime.utcnow),
)

rag_chunks = sa.Table(
    "rag_chunks",
    metadata,
    sa.Column("id", sa.Integer(), primary_key=True),
    sa.Column("document_id", sa.Integer(), sa.ForeignKey("rag_documents.id", ondelete="CASCADE")),
    sa.Column("chunk_index", sa.Integer(), nullable=False),
    sa.Column("content", sa.Text(), nullable=False),
    sa.Column("embedding", sa.Text()),
)


# ---------- CRUD helpers ---------------------------------------------------------

class DocRow:
    def __init__(self, row):
        self.id = row.id
        self.filename = row.filename
        self.file_type = row.file_type
        self.char_count = row.char_count
        self.created_at = row.created_at


class ChunkRow:
    def __init__(self, row):
        self.id = row.id
        self.document_id = row.document_id
        self.chunk_index = row.chunk_index
        self.content = row.content
        self.embedding = row.embedding


def save_document(session: Session, filename: str, file_type: str, char_count: int) -> int:
    result = session.execute(
        rag_documents.insert().values(
            filename=filename,
            file_type=file_type,
            char_count=char_count,
            created_at=datetime.utcnow(),
        )
    )
    session.commit()
    return result.inserted_primary_key[0]


def save_chunks(session: Session, document_id: int, chunks: list[str], embeddings: list[list[float]]) -> None:
    rows = [
        {"document_id": document_id, "chunk_index": i, "content": text, "embedding": json.dumps(vec)}
        for i, (text, vec) in enumerate(zip(chunks, embeddings))
    ]
    if rows:
        session.execute(rag_chunks.insert(), rows)
    session.commit()


def list_documents(session: Session) -> list[DocRow]:
    rows = session.execute(rag_documents.select().order_by(rag_documents.c.created_at.desc())).fetchall()
    return [DocRow(r) for r in rows]


def get_document(session: Session, doc_id: int) -> DocRow | None:
    row = session.execute(rag_documents.select().where(rag_documents.c.id == doc_id)).fetchone()
    return DocRow(row) if row else None


def delete_document(session: Session, doc_id: int) -> bool:
    result = session.execute(rag_documents.delete().where(rag_documents.c.id == doc_id))
    session.commit()
    return result.rowcount > 0


def get_all_chunks(session: Session) -> list[ChunkRow]:
    rows = session.execute(rag_chunks.select()).fetchall()
    return [ChunkRow(r) for r in rows]
