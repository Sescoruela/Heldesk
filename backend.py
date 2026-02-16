"""
backend.py — Motor RAG completo para el Helpdesk interno.

Responsabilidades:
  1. Ingestión de documentos (PDF, DOCX, TXT)
  2. Generación de embeddings vía API de xAI (Grok)
  3. Almacenamiento y búsqueda en FAISS (vector store)
  4. Recuperación semántica de fragmentos relevantes
  5. Generación de respuesta estructurada vía API de Grok (LLM)
"""

from __future__ import annotations

import os
import hashlib
import json
import textwrap
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import faiss
from openai import OpenAI

# ──────────────────────────────────────────────
# Extractores de texto por tipo de archivo
# ──────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extrae texto plano de un archivo PDF."""
    from PyPDF2 import PdfReader
    from io import BytesIO

    reader = PdfReader(BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extrae texto plano de un archivo DOCX."""
    from docx import Document
    from io import BytesIO

    doc = Document(BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs)


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Decodifica un archivo de texto plano."""
    return file_bytes.decode("utf-8", errors="replace")


EXTRACTORS = {
    ".pdf":  extract_text_from_pdf,
    ".docx": extract_text_from_docx,
    ".txt":  extract_text_from_txt,
    ".md":   extract_text_from_txt,
    ".log":  extract_text_from_txt,
}


# ──────────────────────────────────────────────
# Embeddings vía API de xAI (Grok)
# ──────────────────────────────────────────────

XAI_BASE_URL = "https://api.x.ai/v1"
EMBEDDING_MODEL = "v3"


def get_embeddings(texts: List[str], api_key: str) -> np.ndarray:
    """
    Genera embeddings para una lista de textos usando la API de xAI.
    Retorna un array numpy de shape (n_texts, dimension).
    """
    client = OpenAI(api_key=api_key, base_url=XAI_BASE_URL)

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    embeddings = [item.embedding for item in response.data]
    return np.array(embeddings, dtype="float32")


# ──────────────────────────────────────────────
# Chunking de documentos
# ──────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Divide un texto largo en fragmentos de tamaño fijo con solapamiento.
    Esto mejora la cobertura semántica en la búsqueda.
    """
    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


# ──────────────────────────────────────────────
# Knowledge Base (Vector Store)
# ──────────────────────────────────────────────

class KnowledgeBase:
    """
    Base de conocimiento vectorial que gestiona:
      - Ingestión y chunking de documentos
      - Generación de embeddings vía API de xAI
      - Índice FAISS para búsqueda por similitud
      - Persistencia en disco (opcional)
    """

    PERSIST_DIR = Path("kb_store")

    def __init__(self):
        # Almacén de chunks y metadatos
        self.chunks: List[str] = []
        self.metadata: List[Dict[str, str]] = []  # {"source": filename}
        self.doc_hashes: set[str] = set()

        # Índice FAISS (se crea al añadir el primer documento)
        self.index: faiss.IndexFlatIP | None = None
        self.dimension: int | None = None

        # Intentar cargar estado persistido
        self._load()

    # --- Ingestión ---

    def add_document(self, filename: str, file_bytes: bytes, api_key: str) -> int:
        """
        Ingesta un documento: extrae texto, lo divide en chunks,
        genera embeddings vía API y los añade al índice FAISS.
        Retorna el número de chunks añadidos.
        """
        # Evitar duplicados por hash de contenido
        doc_hash = hashlib.sha256(file_bytes).hexdigest()
        if doc_hash in self.doc_hashes:
            return 0

        ext = Path(filename).suffix.lower()
        extractor = EXTRACTORS.get(ext)
        if extractor is None:
            raise ValueError(
                f"Formato '{ext}' no soportado. "
                f"Formatos válidos: {', '.join(EXTRACTORS.keys())}"
            )

        raw_text = extractor(file_bytes)
        if not raw_text.strip():
            return 0

        new_chunks = chunk_text(raw_text)
        if not new_chunks:
            return 0

        # Generar embeddings vía API de xAI
        embeddings = get_embeddings(new_chunks, api_key)

        # Normalizar para similitud coseno
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms

        # Inicializar el índice si es la primera vez
        if self.index is None:
            self.dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(self.dimension)

        # Añadir al índice y registros internos
        self.index.add(embeddings)
        for chunk in new_chunks:
            self.chunks.append(chunk)
            self.metadata.append({"source": filename})

        self.doc_hashes.add(doc_hash)
        self._save()
        return len(new_chunks)

    # --- Búsqueda semántica ---

    def search(self, query: str, api_key: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Busca los fragmentos más similares a la consulta.
        Retorna una lista de dicts con keys: chunk, source, score.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        # Generar embedding de la consulta vía API
        query_emb = get_embeddings([query], api_key)
        norms = np.linalg.norm(query_emb, axis=1, keepdims=True)
        norms[norms == 0] = 1
        query_emb = query_emb / norms

        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_emb, top_k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append({
                "chunk":  self.chunks[idx],
                "source": self.metadata[idx]["source"],
                "score":  float(score),
            })
        return results

    # --- Estadísticas ---

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    @property
    def total_documents(self) -> int:
        return len(self.doc_hashes)

    @property
    def sources(self) -> List[str]:
        seen: set[str] = set()
        out: List[str] = []
        for m in self.metadata:
            if m["source"] not in seen:
                seen.add(m["source"])
                out.append(m["source"])
        return out

    # --- Resetear la base ---

    def reset(self):
        """Elimina todos los documentos y el índice."""
        self.chunks.clear()
        self.metadata.clear()
        self.doc_hashes.clear()
        self.index = None
        self.dimension = None
        if self.PERSIST_DIR.exists():
            import shutil
            shutil.rmtree(self.PERSIST_DIR)

    # --- Persistencia sencilla en disco ---

    def _save(self):
        self.PERSIST_DIR.mkdir(exist_ok=True)
        # Guardar metadatos y chunks
        data = {
            "chunks":     self.chunks,
            "metadata":   self.metadata,
            "doc_hashes": list(self.doc_hashes),
        }
        (self.PERSIST_DIR / "data.json").write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )
        # Guardar índice FAISS
        if self.index is not None:
            faiss.write_index(self.index, str(self.PERSIST_DIR / "index.faiss"))

    def _load(self):
        data_path = self.PERSIST_DIR / "data.json"
        index_path = self.PERSIST_DIR / "index.faiss"
        if data_path.exists() and index_path.exists():
            data = json.loads(data_path.read_text(encoding="utf-8"))
            self.chunks = data["chunks"]
            self.metadata = data["metadata"]
            self.doc_hashes = set(data["doc_hashes"])
            self.index = faiss.read_index(str(index_path))
            self.dimension = self.index.d


# ──────────────────────────────────────────────
# Generación de respuesta con Grok (LLM)
# ──────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
    Eres un asistente experto de Helpdesk interno de TI.
    Tu trabajo es resolver incidencias técnicas de manera clara y profesional.

    REGLAS:
    - Responde EXCLUSIVAMENTE con la información proporcionada en el CONTEXTO.
    - Si el contexto no contiene información suficiente, indícalo con honestidad.
    - Estructura tu respuesta SIEMPRE con las siguientes secciones en Markdown:

    ## 🔍 Diagnóstico
    (Resumen del problema identificado)

    ## 📋 Pasos para resolver
    (Lista numerada de acciones concretas)

    ## ✅ Checklist de verificación
    (Lista de comprobaciones para confirmar la resolución)

    ## 📚 Fuentes consultadas
    (Documentos de donde se extrajo la información)
""")


def build_user_prompt(query: str, context_chunks: List[Dict[str, Any]]) -> str:
    """Construye el prompt de usuario con el contexto recuperado."""
    if not context_chunks:
        context_block = "(No se encontró contexto relevante en la base de conocimiento.)"
    else:
        parts: list[str] = []
        for i, c in enumerate(context_chunks, 1):
            parts.append(
                f"--- Fragmento {i} (Fuente: {c['source']}, "
                f"Relevancia: {c['score']:.2f}) ---\n{c['chunk']}"
            )
        context_block = "\n\n".join(parts)

    return (
        f"CONTEXTO:\n{context_block}\n\n"
        f"INCIDENCIA REPORTADA:\n{query}\n\n"
        "Proporciona una respuesta estructurada siguiendo el formato indicado."
    )


def generate_response(
    query: str,
    context_chunks: List[Dict[str, Any]],
    api_key: str,
    model: str = "grok-3-mini",
) -> str:
    """
    Llama a la API de Grok para generar la respuesta final
    sintetizando la información recuperada.
    """
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1",
    )

    user_prompt = build_user_prompt(query, context_chunks)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=2048,
    )

    return response.choices[0].message.content


# ──────────────────────────────────────────────
# Orquestador RAG (pipeline completo)
# ──────────────────────────────────────────────

def rag_pipeline(
    query: str,
    kb: KnowledgeBase,
    api_key_embeddings: str,
    api_key_llm: str,
    top_k: int = 5,
    model: str = "grok-3-mini",
) -> Dict[str, Any]:
    """
    Pipeline RAG completo:
      1. Recupera fragmentos relevantes del vector store
      2. Genera la respuesta con Grok
      3. Retorna respuesta + fuentes utilizadas
    """
    # 1. Recuperación semántica
    results = kb.search(query, api_key=api_key_embeddings, top_k=top_k)

    # 2. Generación con LLM
    answer = generate_response(query, results, api_key_llm, model=model)

    # 3. Fuentes únicas
    sources = list({r["source"] for r in results})

    return {
        "answer":  answer,
        "sources": sources,
        "context": results,
    }
