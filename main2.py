"""Select a previously processed PDF and ask questions about its graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from kg_retrieve.qa_generator import MODEL_NAME, run_qa

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Data"


def available_documents() -> list[str]:
    if not DATA_DIR.is_dir():
        return []
    return sorted(
        folder.name for folder in DATA_DIR.iterdir()
        if folder.is_dir() and (folder / "knowledge_graph.json").is_file()
        and document_complete(folder)
    )


def document_complete(folder: Path) -> bool:
    """Require completion only when a manifest is available."""
    manifest = folder / "pipeline_manifest.json"
    if not manifest.is_file():
        return True  # Existing manually imported graph; no pipeline status to read
    try:
        return json.loads(manifest.read_text(encoding="utf-8")).get("status") == "complete"
    except (json.JSONDecodeError, OSError):
        return False


def select_graph(typed_name: str) -> Path:
    supplied = typed_name.strip().strip('"').strip("'")
    if not supplied:
        raise ValueError("Enter the PDF name associated with the graph")
    # Accept `report`, `report.pdf`, or the folder name. Do not interpret user
    # input as a file-system path outside the Data directory.
    document = supplied[:-4] if supplied.casefold().endswith(".pdf") else supplied
    if not document or Path(document).name != document or document in {".", ".."}:
        raise ValueError("Enter the PDF filename, not a path")
    folder = DATA_DIR / document
    graph = folder / "knowledge_graph.json"
    if not graph.is_file():
        choices = ", ".join(available_documents()) or "none"
        raise FileNotFoundError(
            f"No knowledge graph for {document!r}. Available documents: {choices}. "
            "Run main1.py for that PDF first."
        )
    if not document_complete(folder):
        raise RuntimeError(
            f"The most recent build for {document!r} is incomplete or failed. "
            "Run main1.py again to complete it."
        )
    return graph


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Ask questions using a selected PDF's graph")
    parser.add_argument("--pdf", help="Previously processed PDF filename")
    parser.add_argument("--raw", action="store_true", help="Display retrieval evidence before LLM calls")
    parser.add_argument("--model", default=MODEL_NAME, help="NVIDIA chat model name")
    args = parser.parse_args(argv)
    if not args.pdf:
        choices = available_documents()
        if choices:
            print("Available PDFs: " + ", ".join(name + ".pdf" for name in choices))
        else:
            print("No completed knowledge graphs found in Data/. Run main1.py first.")
        try:
            args.pdf = input("Enter PDF name for Q&A (example: my_document.pdf): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nNo PDF selected.", file=sys.stderr)
            return 1
    try:
        graph = select_graph(args.pdf)
        run_qa(graph, model=args.model, show_raw=args.raw)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
