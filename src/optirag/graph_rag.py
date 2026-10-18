from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Literal, TypedDict

from dotenv import load_dotenv
from langchain_chroma import Chroma
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

load_dotenv()
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PDF_PATH = ROOT / "data" / "Stock_Market_Performance_2024.pdf"
PERSIST_DIR = ROOT / "chroma_db"
COLLECTION = "stock_market"
MAX_RETRIES = 2


class RAGState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    retries: int


def load_pdf_pages(path: Path) -> List[Document]:
    """Load a PDF into one Document per non-empty page."""
    reader = PdfReader(str(path))
    docs: List[Document] = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(
                Document(page_content=text, metadata={"source": str(path), "page": i})
            )
    return docs


def build_vectorstore(embeddings: HuggingFaceEmbeddings) -> Chroma:
    """Create or reuse the persisted Chroma store (no duplicate ingestion)."""
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
    )
    if store.get(limit=1)["ids"]:
        logger.info("Reusing existing collection")
        return store

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    pages = load_pdf_pages(PDF_PATH)
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(pages)
    store.add_documents(chunks)
    logger.info("Ingested %d pages -> %d chunks", len(pages), len(chunks))
    return store


def build_graph(llm: ChatGroq, retriever):
    def retrieve(state: RAGState) -> dict:
        docs = retriever.invoke(state["question"])
        return {"documents": docs, "retries": state.get("retries", 0)}

    def grade_documents(state: RAGState) -> Literal["generate", "rewrite"]:
        docs = state["documents"]
        if not docs:
            return "rewrite"
        content = "\n\n".join(d.page_content for d in docs)
        prompt = (
            f"Question: {state['question']}\n\nDocuments: {content}\n\n"
            "Are these documents relevant to answering the question?\n"
            "Return only YES or NO."
        )
        verdict = llm.invoke(prompt).content.strip().upper()
        return "generate" if verdict.startswith("YES") else "rewrite"

    def rewrite_question(state: RAGState) -> dict:
        prompt = (
            "Improve the search query for the annual financial report.\n"
            "Preserve its intent and make it specific.\n\n"
            f"Query: {state['question']}\nReturn only the revised query."
        )
        return {
            "question": llm.invoke(prompt).content.strip(),
            "retries": state.get("retries", 0) + 1,
        }

    def generate_answer(state: RAGState) -> dict:
        context = "\n\n".join(
            (
                f"Page {d.metadata['page'] + 1}:\n{d.page_content}"
                if isinstance(d.metadata.get("page"), int)
                else f"Page N/A:\n{d.page_content}"
            )
            for d in state["documents"]
        )
        prompt = (
            "You are a stock market financial analyst.\n"
            "Answer using only the annual report context. Cite page numbers. "
            "Never invent figures or predict returns. "
            "State when evidence is insufficient.\n\n"
            f"Question: {state['question']}\n\nContext:\n{context}"
        )
        return {"generation": llm.invoke(prompt).content}

    def after_rewrite(state: RAGState) -> Literal["retrieve", "generate"]:
        return "retrieve" if state["retries"] < MAX_RETRIES else "generate"

    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve)
    g.add_node("rewrite", rewrite_question)
    g.add_node("generate", generate_answer)
    g.add_edge(START, "retrieve")
    g.add_conditional_edges(
        "retrieve", grade_documents, {"generate": "generate", "rewrite": "rewrite"}
    )
    g.add_conditional_edges(
        "rewrite", after_rewrite, {"retrieve": "retrieve", "generate": "generate"}
    )
    g.add_edge("generate", END)
    return g.compile()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    retriever = build_vectorstore(embeddings).as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )
    app = build_graph(llm, retriever)

    question = (
        "Analyze the company's financial performance, revenue and profit growth, "
        "debt, cash flow, key risks, future outlook, and shareholder returns."
    )
    result = app.invoke(
        {"question": question, "documents": [], "generation": "", "retries": 0}
    )
    print(result["generation"])


if __name__ == "__main__":
    main()
