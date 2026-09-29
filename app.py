"""
University Student & Academic Knowledge Assistant
--------------------------------------------------
RAG application:
User question
    -> BGE-M3 query embedding
    -> FAISS cosine-similarity retrieval
    -> metadata/source reconstruction
    -> Groq GPT-OSS 120B
    -> grounded answer + traceable sources

IMPORTANT:
- Original PDFs are NOT required by this application.
- Only the pre-built FAISS index and metadata are loaded.
- The query embedding model MUST remain BAAI/bge-m3 because the
  stored FAISS vectors were created with that model.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer


# ================================================================
# CONFIGURATION
# ================================================================

APP_TITLE = "University Student & Academic Knowledge Assistant"

BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "rag_index"

INDEX_FILE = INDEX_DIR / "index.faiss"
METADATA_FILE = INDEX_DIR / "metadata.json"
EMBEDDING_CONFIG_FILE = INDEX_DIR / "embedding_config.json"
MANIFEST_FILE = INDEX_DIR / "document_manifest.json"

EXPECTED_EMBEDDING_MODEL = "BAAI/bge-m3"
GROQ_MODEL = "openai/gpt-oss-120b"

DEFAULT_TOP_K = 5
MAX_TOP_K = 10

SYSTEM_PROMPT = """
You are a university student and academic information assistant.

Your task is to answer questions ONLY from the retrieved university
documents supplied in the context.

Rules:
1. Use only the supplied retrieved context as factual evidence.
2. Do not invent university rules, dates, fees, requirements, policies,
   procedures, or exceptions.
3. If the retrieved context does not contain enough information, clearly
   say that the available documents do not provide enough information.
4. When the context contains conflicting information, identify the conflict
   and cite the relevant sources rather than silently choosing one.
5. Keep answers clear, concise, and useful for students.
6. Preserve important conditions, exceptions, thresholds, deadlines,
   eligibility requirements, and procedural steps when they are present.
7. Every substantive claim should have an inline source citation in this
   format: [Document Name, p. X].
8. Do not cite a source that does not support the statement.
9. At the end, include a short "Sources" section listing the documents/pages
   actually used.
10. Do not mention FAISS, embeddings, retrieval, or internal implementation
    unless the user asks about the system itself.
""".strip()


# ================================================================
# PAGE CONFIG / STYLING
# ================================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .source-card {
        border: 1px solid rgba(128,128,128,.30);
        border-radius: 10px;
        padding: 12px 14px;
        margin: 8px 0;
    }
    .source-title {
        font-weight: 700;
        margin-bottom: 4px;
    }
    .source-meta {
        font-size: 0.88rem;
        opacity: 0.78;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ================================================================
# RESOURCE LOADING
# ================================================================

@st.cache_resource(show_spinner=False)
def load_rag_resources() -> tuple[Any, dict, dict, list]:
    """Load FAISS, metadata, embedding configuration and manifest once."""

    required_files = [
        INDEX_FILE,
        METADATA_FILE,
        EMBEDDING_CONFIG_FILE,
        MANIFEST_FILE,
    ]

    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing RAG index files:\n" + "\n".join(missing)
        )

    index = faiss.read_index(str(INDEX_FILE))

    with METADATA_FILE.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    with EMBEDDING_CONFIG_FILE.open("r", encoding="utf-8") as f:
        embedding_config = json.load(f)

    with MANIFEST_FILE.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    stored_model = embedding_config.get("embedding_model")
    if stored_model != EXPECTED_EMBEDDING_MODEL:
        raise RuntimeError(
            "Embedding model mismatch. "
            f"Index was created with '{stored_model}', but the app expects "
            f"'{EXPECTED_EMBEDDING_MODEL}'."
        )

    records = metadata.get("records", [])

    if index.ntotal != len(records):
        raise RuntimeError(
            f"FAISS/metadata mismatch: FAISS has {index.ntotal} vectors, "
            f"but metadata contains {len(records)} records."
        )

    expected_dimension = embedding_config.get("embedding_dimension")
    if expected_dimension and index.d != expected_dimension:
        raise RuntimeError(
            f"Embedding dimension mismatch: FAISS={index.d}, "
            f"metadata={expected_dimension}."
        )

    return index, metadata, embedding_config, manifest


@st.cache_resource(show_spinner="Loading BGE-M3 embedding model...")
def load_embedding_model() -> SentenceTransformer:
    """
    Load the same embedding model used by the Colab indexing notebook.

    CPU is deliberately used for broad Streamlit Community Cloud
    compatibility. The model is cached after the first load.
    """
    return SentenceTransformer(
        EXPECTED_EMBEDDING_MODEL,
        device="cpu",
    )


@st.cache_resource(show_spinner=False)
def load_groq_client() -> Groq:
    """Create the Groq client from the environment/Streamlit secret."""

    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            api_key = None

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. "
            "Add it to Streamlit Secrets or as an environment variable."
        )

    return Groq(api_key=api_key)


# ================================================================
# RETRIEVAL
# ================================================================

def embed_query(
    model: SentenceTransformer,
    query: str,
) -> np.ndarray:
    """Embed one query using the same model/normalization as indexing."""

    vector = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return np.asarray(vector, dtype=np.float32)


def retrieve(
    index: Any,
    records: list[dict],
    model: SentenceTransformer,
    query: str,
    top_k: int,
) -> list[dict]:
    """Retrieve top-k chunks and attach source metadata."""

    query_vector = embed_query(model, query)

    scores, indices = index.search(query_vector, top_k)

    results: list[dict] = []

    for rank, (score, vector_index) in enumerate(
        zip(scores[0], indices[0]),
        start=1,
    ):
        vector_index = int(vector_index)

        if vector_index < 0 or vector_index >= len(records):
            continue

        record = records[vector_index]
        metadata = record.get("metadata", {})

        results.append(
            {
                "rank": rank,
                "score": float(score),
                "vector_index": vector_index,
                "chunk_id": metadata.get(
                    "chunk_id",
                    record.get("id"),
                ),
                "text": record.get("text", ""),
                "source_file": metadata.get(
                    "source_file",
                    "Unknown document",
                ),
                "page_number": metadata.get(
                    "page_number",
                    "Unknown",
                ),
                "total_pages": metadata.get(
                    "total_pages",
                    "Unknown",
                ),
                "chunk_number_on_page": metadata.get(
                    "chunk_number_on_page",
                    "Unknown",
                ),
                "chunk_characters": metadata.get(
                    "chunk_characters",
                    len(record.get("text", "")),
                ),
                "document_type": metadata.get(
                    "document_type",
                    "PDF",
                ),
            }
        )

    return results


# ================================================================
# CONTEXT CONSTRUCTION
# ================================================================

def build_context(results: list[dict]) -> str:
    """Build a clearly delimited, source-aware context for the LLM."""

    blocks = []

    for result in results:
        source = (
            f"{result['source_file']}, "
            f"p. {result['page_number']}"
        )

        block = (
            f"--- SOURCE {result['rank']} ---\n"
            f"Document: {result['source_file']}\n"
            f"Page: {result['page_number']} "
            f"of {result['total_pages']}\n"
            f"Chunk ID: {result['chunk_id']}\n"
            f"Source citation: [{source}]\n"
            f"Retrieved similarity: {result['score']:.4f}\n"
            f"Text:\n{result['text']}"
        )

        blocks.append(block)

    return "\n\n".join(blocks)


# ================================================================
# GROQ GENERATION
# ================================================================

def generate_answer(
    client: Groq,
    question: str,
    context: str,
) -> str:
    """Generate a grounded response with Groq GPT-OSS 120B."""

    user_prompt = f"""
Answer the following university-related question using ONLY the
retrieved document context below.

QUESTION:
{question}

RETRIEVED DOCUMENT CONTEXT:
{context}

Remember:
- Do not use outside knowledge.
- Cite substantive claims as [Document Name, p. X].
- If the context is insufficient, say so explicitly.
- Do not fabricate missing information.
""".strip()

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        model=GROQ_MODEL,
        temperature=0.1,
        max_completion_tokens=1800,
    )

    content = chat_completion.choices[0].message.content

    if not content:
        raise RuntimeError("Groq returned an empty response.")

    return content.strip()


# ================================================================
# SOURCE DISPLAY
# ================================================================

def show_sources(results: list[dict]) -> None:
    """Display traceable source information below the answer."""

    st.subheader("Retrieved Sources")

    for result in results:
        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-title">
                    {result["source_file"]}
                </div>
                <div class="source-meta">
                    Page {result["page_number"]} of {result["total_pages"]}
                    &nbsp; | &nbsp;
                    Chunk {result["chunk_id"]}
                    &nbsp; | &nbsp;
                    Similarity {result["score"]:.4f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander(
            f"View retrieved text — {result['source_file']} "
            f"(page {result['page_number']})"
        ):
            st.write(result["text"])


# ================================================================
# APPLICATION
# ================================================================

st.title("🎓 University Student & Academic Knowledge Assistant")
st.caption(
    "Answers are generated from the indexed university documents "
    "with document and page-level source traceability."
)

try:
    index, metadata_payload, embedding_config, manifest = load_rag_resources()
    embedding_model = load_embedding_model()
except Exception as exc:
    st.error("The RAG system could not be initialized.")
    st.exception(exc)
    st.stop()

with st.sidebar:
    st.header("RAG Configuration")

    top_k = st.slider(
        "Retrieved chunks",
        min_value=3,
        max_value=MAX_TOP_K,
        value=DEFAULT_TOP_K,
        step=1,
    )

    st.divider()

    st.write("**Embedding model**")
    st.code(embedding_config.get("embedding_model", EXPECTED_EMBEDDING_MODEL))

    st.write("**Vector database**")
    st.code("FAISS IndexFlatIP")

    st.write("**Similarity**")
    st.code("Cosine similarity")

    st.write("**LLM**")
    st.code(GROQ_MODEL)

    st.divider()

    st.write(
        f"**Indexed documents:** "
        f"{len(manifest)}"
    )

    st.write(
        f"**Indexed vectors:** "
        f"{index.ntotal}"
    )

    st.caption(
        "Original PDF files are not required at runtime. "
        "Source traceability comes from metadata.json."
    )

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):
            with st.expander("Sources used"):
                show_sources(message["sources"])


question = st.chat_input(
    "Ask a question about university rules, policies, fees, admissions, examinations, scholarships, etc."
)

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.status(
                "Searching the university knowledge base...",
                expanded=False,
            ) as status:
                results = retrieve(
                    index=index,
                    records=metadata_payload["records"],
                    model=embedding_model,
                    query=question,
                    top_k=top_k,
                )

                if not results:
                    status.update(
                        label="No relevant information found.",
                        state="error",
                    )
                    answer = (
                        "I could not find relevant information in the "
                        "indexed university documents."
                    )
                    st.markdown(answer)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": [],
                        }
                    )
                    st.stop()

                status.update(
                    label=f"Retrieved {len(results)} relevant chunks.",
                    state="complete",
                )

            with st.spinner("Generating a grounded answer..."):
                groq_client = load_groq_client()
                context = build_context(results)
                answer = generate_answer(
                    client=groq_client,
                    question=question,
                    context=context,
                )

            st.markdown(answer)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": results,
                }
            )

            with st.expander("Sources used"):
                show_sources(results)

        except Exception as exc:
            error_message = (
                "An error occurred while processing the question. "
                "Please check the application configuration."
            )
            st.error(error_message)
            st.exception(exc)
