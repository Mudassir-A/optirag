from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Literal, Optional, TypedDict

import chromadb
import numpy as np
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from pypdf import PdfReader

load_dotenv()
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PDF_PATH = Path(os.getenv("OPTIRAG_PDF_PATH", ROOT / "data" / "document.pdf"))
CHROMA_DIR = ROOT / "chroma_db"
CACHE_DIR = ROOT / "cache_db"
LOG_DIR = ROOT / "logs"

SMALL_MODEL = os.getenv("OPTIRAG_SMALL_MODEL", "openai/gpt-oss-20b")
LARGE_MODEL = os.getenv("OPTIRAG_LARGE_MODEL", "openai/gpt-oss-120b")
EMBED_MODEL = os.getenv("OPTIRAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CACHE_THRESHOLD = float(os.getenv("OPTIRAG_CACHE_THRESHOLD", "0.90"))
CACHE_TTL_HOURS = float(os.getenv("OPTIRAG_CACHE_TTL_HOURS", "168"))
ROUTER_THRESHOLD = float(os.getenv("OPTIRAG_ROUTER_THRESHOLD", "2.0"))
TOP_K = int(os.getenv("OPTIRAG_RETRIEVAL_K", "3"))
CHUNK_SIZE, CHUNK_OVERLAP, MAX_RETRIES = 1000, 200, 2


# ----------------------------------------------------------------- logging
def setup_logging() -> None:
    """Send logs to logs/optirag.log only; the terminal shows just the answer."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        LOG_DIR / "optirag.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    for noisy in ("httpx", "httpcore", "chromadb", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# --------------------------------------------------------------- ingestion
def load_pdf_pages(path: Path) -> List[Document]:
    reader = PdfReader(str(path))
    docs = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(Document(page_content=text, metadata={"source": str(path), "page": i}))
    return docs


def corpus_version(path: Path) -> str:
    """Hash of PDF bytes + chunking params; changes whenever the corpus changes."""
    digest = hashlib.sha256(path.read_bytes())
    digest.update(f"|{CHUNK_SIZE}|{CHUNK_OVERLAP}".encode())
    return digest.hexdigest()[:12]


def build_vectorstore(embeddings: HuggingFaceEmbeddings, path: Path, version: str) -> Chroma:
    """One collection per corpus version, so different PDFs never mix."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    store = Chroma(
        collection_name=f"docs_{version}",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    if store.get(limit=1)["ids"]:
        return store
    pages = load_pdf_pages(path)
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(pages)
    store.add_documents(chunks)
    logger.info("Ingested %d pages -> %d chunks", len(pages), len(chunks))
    return store


# ---------------------------------------------------------- semantic cache
class SemanticCache:
    """Embeds questions, returns a stored answer when cosine similarity >= threshold.

    Entries are tagged with corpus_version and a timestamp; lookups only consider
    entries from the current corpus that are within the TTL.
    """

    def __init__(self, embed_fn, version: str, threshold: float, ttl_seconds: float) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._embed_fn = embed_fn
        self._version = version
        self._threshold = threshold
        self._ttl = ttl_seconds
        self._client = chromadb.PersistentClient(path=str(CACHE_DIR))
        self._col = self._client.get_or_create_collection("query_cache")

    def _embed(self, text: str) -> np.ndarray:
        v = np.asarray(self._embed_fn(text), dtype=np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def lookup(self, question: str) -> tuple[Optional[str], float]:
        """Return (answer or None, best similarity among valid entries)."""
        if self._col.count() == 0:
            return None, 0.0
        q = self._embed(question)
        res = self._col.query(
            query_embeddings=[q.tolist()],
            n_results=1,
            where={
                "$and": [
                    {"corpus_version": {"$eq": self._version}},
                    {"created_at": {"$gte": time.time() - self._ttl}},
                ]
            },
            include=["documents", "embeddings"],
        )
        if not res["ids"] or not res["ids"][0]:
            return None, 0.0
        cand = np.asarray(res["embeddings"][0][0], dtype=np.float32)
        norm = float(np.linalg.norm(cand))
        score = float(np.dot(q, cand / norm)) if norm > 0 else 0.0
        return (res["documents"][0][0] if score >= self._threshold else None), score

    def store(self, question: str, answer: str) -> None:
        key = hashlib.sha1(f"{self._version}|{question.strip().lower()}".encode()).hexdigest()
        self._col.upsert(
            ids=[key],
            embeddings=[self._embed(question).tolist()],
            documents=[answer],
            metadatas=[
                {"question": question, "corpus_version": self._version, "created_at": time.time()}
            ],
        )

    def invalidate_stale(self) -> int:
        """Delete entries from other corpus versions or past their TTL."""
        before = self._col.count()
        if before:
            self._col.delete(
                where={
                    "$or": [
                        {"corpus_version": {"$ne": self._version}},
                        {"created_at": {"$lt": time.time() - self._ttl}},
                    ]
                }
            )
        return before - self._col.count()

    def clear(self) -> None:
        self._client.delete_collection("query_cache")
        self._col = self._client.get_or_create_collection("query_cache")


# ------------------------------------------------------------------ routing
ANALYTIC = re.compile(
    r"\b(analy[sz]e|analysis|evaluate|assess|summari[sz]e|explain why|why|implications?|"
    r"trade-?offs?|relationship|impact|overall|in depth|step[- ]by[- ]step)\b",
    re.I,
)
COMPARATIVE = re.compile(r"\b(compare|contrast|versus|vs\.?|difference between|pros and cons)\b", re.I)
INSUFFICIENT = re.compile(
    r"(insufficient (evidence|information|context)|"
    r"not (enough|sufficient) (evidence|information|context)|"
    r"cannot (be )?(determine|answer|find)|"
    r"does not (contain|provide|mention|include))",
    re.I,
)
CITATION = re.compile(r"\b(?:pages?|pp?\.)\s*\d+", re.I)


def route_query(question: str, threshold: float = ROUTER_THRESHOLD) -> Literal["small", "large"]:
    """Cheap heuristic complexity score; >= threshold goes to the large model."""
    score = 0.0
    words = len(question.split())
    if words >= 40:
        score += 1.5
    elif words >= 20:
        score += 0.75

    analytic = len(ANALYTIC.findall(question))
    score += 1.5 if analytic >= 2 else 1.0 if analytic == 1 else 0.0
    if COMPARATIVE.search(question):
        score += 1.0
    if (
        question.count("?") > 1
        or question.count(",") >= 3
        or len(re.findall(r"\band\b", question, re.I)) >= 3
    ):
        score += 1.0
    return "large" if score >= threshold else "small"


def needs_escalation(answer: str, tier: str, docs_relevant: bool) -> tuple[bool, str]:
    """Should a small-model answer be regenerated on the large model?"""
    if tier == "large":
        return False, "already_large"
    if not docs_relevant:
        return False, "no_relevant_evidence"  # a bigger model can't fix missing evidence
    if len(answer.strip()) < 80:
        return True, "answer_too_short"
    if INSUFFICIENT.search(answer):
        return True, "insufficient_evidence_claimed"
    if not CITATION.search(answer):
        return True, "missing_citations"
    return False, "ok"


def is_cacheable(answer: str, docs_relevant: bool) -> bool:
    """Cache only grounded answers, never 'insufficient evidence' replies."""
    return bool(answer.strip()) and docs_relevant and not INSUFFICIENT.search(answer)


# -------------------------------------------------------------------- graph
class RAGState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    retries: int
    relevant: bool
    tier: str
    escalated: bool
    escalation_reason: str


def _context(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        page = d.metadata.get("page")
        label = f"Page {page + 1}" if isinstance(page, int) else "Page N/A"
        parts.append(f"{label}:\n{d.page_content}")
    return "\n\n".join(parts)


def _answer_prompt(question: str, context: str) -> str:
    return (
        "You are a precise document analyst.\n"
        "Answer using only the context below. Cite page numbers like (Page 4). "
        "Do not invent facts or figures. If the context is insufficient, say so.\n\n"
        f"Question: {question}\n\nContext:\n{context}"
    )


def build_graph(retriever, utility: ChatGroq, tiers: dict[str, ChatGroq]):
    def retrieve(state: RAGState) -> dict:
        return {"documents": retriever.invoke(state["question"])}

    def grade(state: RAGState) -> dict:
        if not state["documents"]:
            return {"relevant": False}
        content = "\n\n".join(d.page_content for d in state["documents"])
        prompt = (
            f"Question: {state['question']}\n\nDocuments: {content}\n\n"
            "Are these documents relevant to answering the question?\n"
            "Return only YES or NO."
        )
        return {"relevant": utility.invoke(prompt).content.strip().upper().startswith("YES")}

    def after_grade(state: RAGState) -> Literal["generate", "rewrite"]:
        return "generate" if state["relevant"] or state["retries"] >= MAX_RETRIES else "rewrite"

    def rewrite(state: RAGState) -> dict:
        prompt = (
            "Improve this search query for retrieving passages from a document.\n"
            "Preserve its intent and make it specific.\n\n"
            f"Query: {state['question']}\nReturn only the revised query."
        )
        new_q = utility.invoke(prompt).content.strip() or state["question"]
        return {"question": new_q, "retries": state["retries"] + 1}

    def generate(state: RAGState) -> dict:
        prompt = _answer_prompt(state["question"], _context(state["documents"]))
        return {"generation": tiers[state["tier"]].invoke(prompt).content.strip()}

    def check(state: RAGState) -> dict:
        _, reason = needs_escalation(state["generation"], state["tier"], state["relevant"])
        return {"escalation_reason": reason}

    def after_check(state: RAGState) -> Literal["escalate", "end"]:
        escalate, _ = needs_escalation(state["generation"], state["tier"], state["relevant"])
        return "escalate" if escalate else "end"

    def escalate(state: RAGState) -> dict:
        logger.info("Escalating to large model: %s", state["escalation_reason"])
        prompt = _answer_prompt(state["question"], _context(state["documents"]))
        return {
            "generation": tiers["large"].invoke(prompt).content.strip(),
            "tier": "large",
            "escalated": True,
        }

    g = StateGraph(RAGState)
    for name, fn in [
        ("retrieve", retrieve), ("grade", grade), ("rewrite", rewrite),
        ("generate", generate), ("check", check), ("escalate", escalate),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade, {"generate": "generate", "rewrite": "rewrite"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", "check")
    g.add_conditional_edges("check", after_check, {"escalate": "escalate", "end": END})
    g.add_edge("escalate", END)
    return g.compile()


# ----------------------------------------------------------------- pipeline
class OptiRAG:
    def __init__(self, pdf_path: Path = PDF_PATH) -> None:
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        if not os.getenv("GROQ_API_KEY"):
            raise EnvironmentError("GROQ_API_KEY is not set (add it to .env)")

        self.version = corpus_version(pdf_path)
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        retriever = build_vectorstore(embeddings, pdf_path, self.version).as_retriever(
            search_type="similarity", search_kwargs={"k": TOP_K}
        )
        small = ChatGroq(model=SMALL_MODEL, temperature=0, max_retries=2)
        large = ChatGroq(model=LARGE_MODEL, temperature=0, max_retries=2)
        self.graph = build_graph(retriever, utility=small, tiers={"small": small, "large": large})
        self.cache = SemanticCache(
            embeddings.embed_query, self.version, CACHE_THRESHOLD, CACHE_TTL_HOURS * 3600
        )
        self.cache.invalidate_stale()

    def ask(self, question: str) -> dict:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty")
        start = time.perf_counter()

        score = 0.0
        try:
            cached, score = self.cache.lookup(question)
            if cached:
                return {
                    "answer": cached, "path": "cache_hit", "tier": "cache", "score": score,
                    "latency_ms": (time.perf_counter() - start) * 1000,
                }
        except Exception:
            logger.exception("Cache lookup failed; treating as miss")

        tier = route_query(question)
        state = self.graph.invoke(
            {
                "question": question, "documents": [], "generation": "", "retries": 0,
                "relevant": False, "tier": tier, "escalated": False, "escalation_reason": "",
            }
        )
        answer = state["generation"]

        if is_cacheable(answer, state["relevant"]):
            try:
                self.cache.store(question, answer)
            except Exception:
                logger.exception("Cache store failed")

        return {
            "answer": answer,
            "path": "escalated" if state["escalated"] else "llm",
            "tier": state["tier"],
            "score": score,
            "latency_ms": (time.perf_counter() - start) * 1000,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?")
    parser.add_argument("--pdf", type=Path, default=PDF_PATH)
    parser.add_argument("--clear-cache", action="store_true")
    args = parser.parse_args()

    setup_logging()
    rag = OptiRAG(args.pdf)
    if args.clear_cache:
        rag.cache.clear()
        print("Cache cleared.")
    if not args.question:
        return
    r = rag.ask(args.question)
    print(r["answer"])
    print(f"\n[{r['path']}] tier={r['tier']} cache_score={r['score']:.3f} latency={r['latency_ms']:.0f}ms")


if __name__ == "__main__":
    main()