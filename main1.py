"""PDF -> chunks -> hybrid ontology -> knowledge graph for one document.

Run from any working directory: python main1.py
Use `python main1.py --help` for non-interactive and tuning options.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

from kg_construct.chunker import process_pdf
from kg_construct.ontology_generator import main as generate_ontology
from kg_construct.kg_generator import main as generate_knowledge_graph

ROOT = Path(__file__).resolve().parent
PDFS_DIR = ROOT / "PDFs"
DATA_DIR = ROOT / "Data"


def document_name(filename: str) -> str:
    """Get a safe folder name without changing ordinary PDF basenames."""
    name = Path(filename).name
    stem = name[:-4] if name.casefold().endswith(".pdf") else name
    clean = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem).strip(" .")
    if clean in ("", ".", ".."):
        raise ValueError("Provide a PDF with a valid filename")
    if clean.upper().split(".")[0] in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        clean = "document_" + clean
    return clean


def resolve_pdf(typed_name: str) -> Path:
    """Find a PDF by absolute/relative path, PDFS_DIR, or project root."""
    raw = typed_name.strip().strip('"').strip("'")
    if not raw:
        raise ValueError("No PDF name was entered")
    entered = Path(raw).expanduser()
    if entered.suffix.casefold() != ".pdf":
        entered = Path(str(entered) + ".pdf")
    if entered.is_absolute():
        candidates = [entered]
    else:
        candidates = [Path.cwd() / entered, PDFS_DIR / entered, ROOT / entered]
    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.casefold() == ".pdf":
            return candidate.resolve()
    # Permit rebuilding from the archived source if the user has removed the
    # original PDF from PDFs/. The old document folder name is preserved below.
    if len(entered.parts) == 1:
        archived = DATA_DIR / document_name(entered.name) / "source.pdf"
        if archived.is_file():
            return archived.resolve()
    searched = "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"PDF not found. Put it in {PDFS_DIR}, enter an absolute path, "
        f"or supply an existing relative path.\n  - {searched}"
    )


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def run_pipeline(args: argparse.Namespace) -> Path:
    source = resolve_pdf(args.pdf)
    if source.name == "source.pdf" and source.parent.parent.resolve() == DATA_DIR.resolve():
        # Rerun of an archived input, e.g. Data/my_document/source.pdf.
        folder = source.parent
    else:
        folder = DATA_DIR / document_name(source.name)
    folder.mkdir(parents=True, exist_ok=True)
    input_copy = folder / "source.pdf"
    manifest_path = folder / "pipeline_manifest.json"
    previous = {}
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            print("[WARN] Previous manifest is unreadable; rebuilding.", flush=True)
    sha = file_hash(source)
    if previous.get("source_sha256") not in (None, sha):
        print("[WARN] The PDF content has changed for this document name. "
              "Existing output will be rebuilt and replaced.", flush=True)
    if source != input_copy.resolve():
        shutil.copy2(source, input_copy)

    print(f"\n[PROJECT] Input PDF: {source}", flush=True)
    print(f"[PROJECT] All outputs: {folder}\n", flush=True)
    manifest = {
        "status": "building", "stage": "chunking",
        "document": document_name(source.name),
        "original_filename": previous.get("original_filename", source.name)
        if source == input_copy.resolve() else source.name,
        "source_sha256": sha,
        "source_pdf": "source.pdf", "created_or_updated_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "device": args.device,
            "review": args.review,
            "no_refine_sparse": args.no_refine_sparse,
        },
    }
    write_json(manifest_path, manifest)

    try:
        print("[1/3] PDF TEXT EXTRACTION AND CHUNKING", flush=True)
        num_chunks = process_pdf(
            input_copy, folder / "chunks", args.chunk_size, args.chunk_overlap
        )
        manifest["chunks"] = num_chunks
        manifest["stage"] = "ontology"
        write_json(manifest_path, manifest)

        print("\n[2/3] HYBRID ONTOLOGY GENERATION", flush=True)
        ontology_args = [
            "--chunks", str(folder / "chunks"),
            "--output", str(folder / "ontology.json"),
            "--candidates-output", str(folder / "ontology_candidates.json"),
            "--coverage-report", str(folder / "ontology_coverage.json"),
            "--cache-dir", str(folder / "ontology_hybrid_cache"),
        ]
        if args.review:
            ontology_args.append("--review")
        if args.no_refine_sparse:
            ontology_args.append("--no-refine-sparse")
        if args.refresh:
            ontology_args.append("--refresh")
        if args.preview_only:
            ontology_args.append("--preview-only")
        result = generate_ontology(ontology_args)
        if result not in (0, None):
            raise RuntimeError(f"Ontology generator returned exit code {result}")
        if args.preview_only:
            manifest.update(status="preview_only", stage="done")
            write_json(manifest_path, manifest)
            print("\n[DONE] Local ontology candidates are available; no graph was built.", flush=True)
            return folder

        manifest["stage"] = "knowledge_graph"
        write_json(manifest_path, manifest)
        print("\n[3/3] KNOWLEDGE GRAPH GENERATION", flush=True)
        kg_args = [
            "--chunks", str(folder / "chunks"),
            "--ontology", str(folder / "ontology.json"),
            "--output", str(folder / "knowledge_graph.json"),
            "--graphml", str(folder / "knowledge_graph.graphml"),
            "--cache-dir", str(folder / "kg_chunks"),
            "--device", args.device,
        ]
        if args.refresh:
            kg_args.append("--refresh")
        generate_knowledge_graph(kg_args)
        graph_path = folder / "knowledge_graph.json"
        if not graph_path.exists():
            raise RuntimeError("KG generator finished without creating knowledge_graph.json")
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        manifest.update(
            status="complete", stage="done",
            entities=len(graph.get("entities", [])),
            relationships=len(graph.get("relationships", [])),
        )
        write_json(manifest_path, manifest)
        print("\n" + "=" * 64, flush=True)
        print(f"[DONE] Graph ready: {graph_path}", flush=True)
        print(f"[DONE] {manifest['entities']} entities / {manifest['relationships']} relationships", flush=True)
        print(f"[NEXT] Run `python main2.py` and enter `{folder.name}.pdf`", flush=True)
        return folder
    except BaseException:
        # main2 refuses to use a graph from a failed or incomplete rebuild.
        manifest["status"] = "failed"
        write_json(manifest_path, manifest)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build an isolated knowledge graph for a selected PDF"
    )
    parser.add_argument("--pdf", help="PDF name (in PDFs/), relative path, or full path")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--review", action="store_true", help="Run optional ontology review call")
    parser.add_argument("--no-refine-sparse", action="store_true",
                        help="Disable evidence-supported sparse attribute refinement")
    parser.add_argument("--refresh", action="store_true", help="Ignore cached NLP/LLM/KG results")
    parser.add_argument("--preview-only", action="store_true", help="Write local candidates only, no API/KG")
    args = parser.parse_args(argv)
    if not args.pdf:
        try:
            args.pdf = input("Enter PDF name (example: my_document.pdf): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nNo PDF selected.", file=sys.stderr)
            return 1
    if args.chunk_size <= 0 or not 0 <= args.chunk_overlap < args.chunk_size:
        parser.error("chunk size must be positive and overlap must be between 0 and chunk_size-1")
    if not args.preview_only and not os.getenv("NVIDIA_API_KEY", "").strip():
        print("[ERROR] NVIDIA_API_KEY is not set. Set it in your shell before running main1.py.",
              file=sys.stderr)
        return 2
    try:
        run_pipeline(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Build stopped. Existing checkpoints can be reused.",
              file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
