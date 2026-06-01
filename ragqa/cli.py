"""Command-line interface for ragqa.

Two subcommands:

    ragqa ingest -i <pdfs/ or file.pdf> -o <index/>
    ragqa ask --index <index/> "the question"

The ingest command runs PDFs through the chunker → embedder → FAISS
index and saves the result. The ask command loads the saved index,
runs retrieval, calls the LLM, and prints the answer with a source
block when citations are present.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ragqa.answer import answer
from ragqa.chunking import chunk_pdf
from ragqa.embedding import Embedder
from ragqa.index import Index
from ragqa.llm import GroqClient
from ragqa.retrieval import DEFAULT_K, DEFAULT_MIN_SCORE, retrieve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ragqa",
        description="Document Q&A assistant with grounded citations.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ─── ingest ───
    ingest = sub.add_parser("ingest", help="build an index from one or more PDFs")
    ingest.add_argument("-i", "--input", required=True, type=Path,
                        help="path to a .pdf file OR a directory of PDFs (searched recursively)")
    ingest.add_argument("-o", "--output", required=True, type=Path,
                        help="directory to save the index to")
    ingest.add_argument("--target-chars", type=int, default=600,
                        help="approximate chars per chunk (default: 600)")
    ingest.add_argument("--overlap-chars", type=int, default=100,
                        help="chars repeated between consecutive chunks (default: 100)")

    # ─── ask ───
    ask = sub.add_parser("ask", help="query an index and print a grounded answer")
    ask.add_argument("question", type=str)
    ask.add_argument("--index", required=True, type=Path,
                     help="path to an index created by `ragqa ingest`")
    ask.add_argument("-k", "--top-k", type=int, default=DEFAULT_K,
                     help=f"chunks retrieved (default: {DEFAULT_K})")
    ask.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE,
                     help=f"abstain if no chunk scores above this (default: {DEFAULT_MIN_SCORE})")
    ask.add_argument("--model", default="llama-3.3-70b-versatile",
                     help="Groq model name")

    args = parser.parse_args(argv)

    if args.cmd == "ingest":
        return _cmd_ingest(args)
    if args.cmd == "ask":
        return _cmd_ask(args)
    return 2


def _cmd_ingest(args) -> int:
    pdfs = _find_pdfs(args.input)
    if not pdfs:
        print(f"error: no PDFs found at {args.input}", file=sys.stderr)
        return 1

    print(f"chunking {len(pdfs)} PDF(s)...", file=sys.stderr)
    chunks = []
    for pdf in pdfs:
        page_chunks = chunk_pdf(
            pdf,
            target_chars=args.target_chars,
            overlap_chars=args.overlap_chars,
        )
        print(f"  {pdf.name}: {len(page_chunks)} chunks", file=sys.stderr)
        chunks.extend(page_chunks)

    if not chunks:
        print("error: no extractable text in those PDFs", file=sys.stderr)
        return 1

    print(f"embedding {len(chunks)} chunks (loads model on first run)...",
          file=sys.stderr)
    embedder = Embedder()
    index = Index.build(chunks, embedder)

    args.output.mkdir(parents=True, exist_ok=True)
    index.save(args.output)
    print(f"saved index to {args.output} ({index.size} chunks)", file=sys.stderr)
    return 0


def _cmd_ask(args) -> int:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print(
            "error: GROQ_API_KEY not set. Free key at https://console.groq.com/keys",
            file=sys.stderr,
        )
        return 2

    embedder = Embedder()
    index = Index.load(args.index)

    retrieved = retrieve(
        args.question,
        index=index,
        embedder=embedder,
        k=args.top_k,
        min_score=args.min_score,
    )

    llm = GroqClient(api_key=api_key, model=args.model)
    result = answer(args.question, retrieved=retrieved, llm=llm)

    print(result.text)
    if result.citations:
        print()
        print("Sources:")
        for i, chunk in enumerate(result.citations, start=1):
            print(f"  [{i}] {chunk.source_file}, page {chunk.page}")
    return 0


def _find_pdfs(path: Path) -> list[Path]:
    """Resolve `path` to a list of PDF file paths.

    If it's a file, return [path] if it's a .pdf.
    If it's a directory, search recursively for *.pdf.
    """
    if path.is_file():
        return [path] if path.suffix.lower() == ".pdf" else []
    if path.is_dir():
        return sorted(path.rglob("*.pdf"))
    return []


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
