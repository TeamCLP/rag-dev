import os
import io
import re
import uuid
from typing import List, Tuple

import gradio as gr
import docx2txt
from pypdf import PdfReader

from sentence_transformers import SentenceTransformer
import numpy as np

try:
    import faiss # type: ignore
except Exception as e:
    raise RuntimeError("FAISS n'est pas installé correctement. Vérifiez requirements.txt et réinstallez.")

import httpx
from openai import OpenAI

# ----------------------------
# Config
# ----------------------------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # rapide, léger
CHUNK_SIZE = 600 # caractères
CHUNK_OVERLAP = 120
TOP_K = 5 # nb de passages récupérés

# Modèles / backends proposés dans l'UI
OLLAMA_MODEL_DEFAULT = "llama3.1"
OPENAI_MODEL_DEFAULT = "gpt-4o-mini"

# Charge le modèle d'embeddings
embedder = SentenceTransformer(EMBED_MODEL_NAME)
index = None # FAISS index
chunks: List[str] = [] # stocke les chunks textuels

# ----------------------------
# Utilitaires documents
# ----------------------------
def read_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")

def read_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            continue
        return "\n".join(texts)
    
demo.launch()