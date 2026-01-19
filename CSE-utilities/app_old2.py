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

# --- FAISS (si dispo) ou fallback scikit-learn ---
FAISS_OK = True
try:
    import faiss  # type: ignore
except Exception:
    FAISS_OK = False
    from sklearn.neighbors import NearestNeighbors

import httpx
from openai import OpenAI

# --- Config ---
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 120
TOP_K = 5

OLLAMA_MODEL_DEFAULT = "llama3.1"
OPENAI_MODEL_DEFAULT = "gpt-4o-mini"

embedder = SentenceTransformer(EMBED_MODEL_NAME)

# Index
faiss_index = None
sk_index = None
emb_matrix = None
chunks: List[str] = []

# --- Lecture fichiers ---
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

def read_docx(file_bytes: bytes) -> str:
    tmp_name = f"/tmp/{uuid.uuid4().hex}.docx"
    with open(tmp_name, "wb") as f:
        f.write(file_bytes)
    text = docx2txt.process(tmp_name) or ""
    try:
        os.remove(tmp_name)
    except Exception:
        pass
    return text

EXT_READERS = {".txt": read_txt, ".pdf": read_pdf, ".docx": read_docx}

# --- Chunking ---
def clean_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def chunk_text(t: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    t = clean_text(t)
    if not t:
        return []
    out = []
    start = 0
    while start < len(t):
        end = min(start + size, len(t))
        out.append(t[start:end])
        start += size - overlap
        if start <= 0:
            break
    return out

# --- Indexation ---
def build_index(all_chunks: List[str]):
    global faiss_index, sk_index, emb_matrix
    if not all_chunks:
        return None
    emb = embedder.encode(all_chunks, convert_to_numpy=True, show_progress_bar=True)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb_norm = emb / np.clip(norms, 1e-12, None)

    if FAISS_OK:
        dim = emb_norm.shape[1]
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(emb_norm.astype("float32"))
        sk_index = None
    else:
        emb_matrix = emb_norm.astype("float32")
        sk_index = NearestNeighbors(metric="cosine")
        sk_index.fit(emb_matrix)
        faiss_index = None
    return True

def search_similar(query: str, k: int = TOP_K) -> List[Tuple[int, float]]:
    if faiss_index is None and sk_index is None:
        return []
    q = embedder.encode([query], convert_to_numpy=True)
    q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)
    q = q.astype("float32")

    if faiss_index is not None:
        D, I = faiss_index.search(q, k)
        return list(zip(I[0].tolist(), D[0].tolist()))
    else:
        dist, idx = sk_index.kneighbors(q, n_neighbors=k)
        sim = (1.0 - dist[0]).tolist()  # approx cos similarity
        return list(zip(idx[0].tolist(), sim))

# --- Backends LLM ---
RAG_SYSTEM = (
    "Tu es un assistant qui répond STRICTEMENT en français. "
    "Utilise UNIQUEMENT les passages fournis si possible. "
    "Si l'information n'est pas présente, dis-le clairement. "
    "Cite brièvement les passages pertinents en fin de réponse."
)

async def chat_ollama(model: str, system: str, user: str) -> str:
    url = "http://127.0.0.1:11434/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": False,
        "options": {"temperature": 0.1},
    }
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        return data.get("message", {}).get("content", "")

async def chat_openai(model: str, system: str, user: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "OPENAI_API_KEY est manquant. Définissez la variable d'environnement ou choisissez Ollama."
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content

def build_prompt(query: str, retrieved: List[str]) -> str:
    context = "\n\n".join([f"[Passage {i+1}] {c}" for i, c in enumerate(retrieved)])
    return (
        f"Contexte :\n{context}\n\n"
        f"Question : {query}\n\n"
        f"Consignes : Réponds de façon concise et structurée. Si tu n'es pas sûr, dis-le."
    )

async def answer_query(query: str, backend: str, model_choice: str):
    hits = search_similar(query, TOP_K)
    retrieved_chunks = [chunks[i] for i, _ in hits] if hits else []
    user_prompt = build_prompt(query, retrieved_chunks)
    if backend == "Ollama (local)":
        model = model_choice or OLLAMA_MODEL_DEFAULT
        reply = await chat_ollama(model, RAG_SYSTEM, user_prompt)
    else:
        model = model_choice or OPENAI_MODEL_DEFAULT
        reply = await chat_openai(model, RAG_SYSTEM, user_prompt)
    refs = (
        "\n\n• Références :\n" + "\n".join([f"- Passage {i+1}" for i in range(len(retrieved_chunks))])
        if retrieved_chunks else "\n\n(Aucune référence disponible : index vide ou question hors périmètre.)"
    )
    return reply + refs

# --- Callbacks UI ---
def reset_index_state():
    global faiss_index, sk_index, emb_matrix, chunks
    faiss_index = None
    sk_index = None
    emb_matrix = None
    chunks = []
    return "Index vidé. Uploadez des documents puis cliquez sur 'Indexer'."

def handle_upload(files):
    if not files:
        return "Aucun fichier reçu."
    texts = []
    for f in files:
        name = f.name
        ext = os.path.splitext(name)[1].lower()
        reader = EXT_READERS.get(ext)
        if not reader:
            continue
        data = f.read()
        try:
            txt = reader(data)
        except Exception as e:
            txt = f"[Erreur lecture {name}: {e}]"
        texts.append(f"\n\n===== {name} =====\n\n" + txt)
    if not texts:
        return "Aucun texte exploitable. Formats acceptés: PDF, DOCX, TXT."
    merged = "\n\n".join(texts)
    return f"{len(merged)} caractères chargés. Cliquez sur 'Indexer'."

def build_chunks_and_index(files):
    global chunks
    if not files:
        return "Veuillez d'abord uploader des documents."
    texts = []
    for f in files:
        name = f.name
        ext = os.path.splitext(name)[1].lower()
        reader = EXT_READERS.get(ext)
        if not reader:
            continue
        data = f.read()
        try:
            txt = reader(data)
        except Exception:
            txt = ""
        texts.append(txt)
    full_text = "\n\n".join(texts)
    chunks = chunk_text(full_text)
    if not chunks:
        return "Aucun contenu indexable."
    build_index(chunks)
    engine = "FAISS" if FAISS_OK else "scikit-learn (fallback)"
    return f"Index ({engine}) créé avec {len(chunks)} passages. Vous pouvez poser des questions."

import asyncio
async def _answer(query, backend, model_choice):
    if not query or len(query.strip()) == 0:
        return "Écrivez une question."
    if faiss_index is None and sk_index is None:
        return "Index vide. Uploadez des documents puis cliquez sur 'Indexer'."
    try:
        return await answer_query(query, backend, model_choice)
    except Exception as e:
        return f"Erreur pendant la génération : {e}"

def ask(query, backend, model_choice):
    return asyncio.run(_answer(query, backend, model_choice))

# --- UI ---
with gr.Blocks(title="RAG Local — Débutant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📚 RAG Local (Débutant)
    1) Uploadez plusieurs documents (PDF / DOCX / TXT)
    2) Cliquez **Indexer**
    3) Posez vos **questions**
    """)
    with gr.Row():
        files = gr.File(label="Documents", file_count="multiple",
                        file_types=[".pdf", ".docx", ".txt"], height=120)
    with gr.Row():
        btn_parse = gr.Button("Indexer", variant="primary")
        btn_reset = gr.Button("Réinitialiser l'index")
    status = gr.Markdown("Prêt.")

    with gr.Accordion("Paramètres IA", open=False):
        backend = gr.Radio(["Ollama (local)", "OpenAI"], value="Ollama (local)", label="Backend")
        model_choice = gr.Textbox(value=OLLAMA_MODEL_DEFAULT, label="Nom du modèle (ex: llama3.1 ou gpt-4o-mini)")
        gr.Markdown("""
        **Ollama (local)** : installez https://ollama.com puis `ollama pull llama3.1`.
        **OpenAI** : définissez `OPENAI_API_KEY` dans votre environnement.
        """)

    query = gr.Textbox(label="Votre question", placeholder="Ex: Quels sont les points clés du contrat ?")
    btn_ask = gr.Button("Poser la question", variant="primary")
    answer = gr.Markdown()

    # Events
    files.upload(fn=handle_upload, inputs=files, outputs=status)
    btn_parse.click(fn=build_chunks_and_index, inputs=files, outputs=status)
    btn_reset.click(fn=reset_index_state, outputs=status)
    btn_ask.click(fn=ask, inputs=[query, backend, model_choice], outputs=answer)

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",   # ou "0.0.0.0" si 127.0.0.1 coince
        server_port=7860,
        share=True,                # force un lien gradio (contourne proxy)
        show_error=True
    )

