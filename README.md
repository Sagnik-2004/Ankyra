# Ankyra

```PDF → Hybrid Ontology → Knowledge Graph → Q&A```

This project integrates the six provided Python modules into **two interactive entry points**:

- **`main1.py`** builds a separate knowledge graph for each PDF by calling `chunker.py`, `ontology_generator.py`, and `kg_generator.py` in sequence.
- **`main2.py`** asks which PDF to query, loads *that PDF's* `knowledge_graph.json`, and runs the existing NetworkX retrieval + NVIDIA Nemotron natural-language answering pipeline.

The original approach is retained: pypdf + LangChain text chunking; spaCy, KeyBERT, MiniLM embeddings, and NVIDIA Nemotron for the document-specific hybrid ontology; local GLiNER2 for entities/relations; NuExtract-1.5-tiny for attributes; and NetworkX + RapidFuzz for graph retrieval.

## Project layout

```text
Ankyra/
├── PDFs/                          # Place input PDF documents here
├── Data/
│   ├── my_document/               # Created automatically after main1.py
│   │   ├── source.pdf              # Copy of the input PDF
│   │   ├── chunks/
│   │   │   ├── chunk_0001.txt
│   │   │   └── ...
│   │   ├── ontology_hybrid_cache/  # Local and NVIDIA ontology checkpoints
│   │   ├── kg_chunks/              # Per-chunk KG checkpoints for resume
│   │   ├── ontology.json
│   │   ├── ontology_candidates.json
│   │   ├── ontology_coverage.json
│   │   ├── knowledge_graph.json
│   │   ├── knowledge_graph.graphml
│   │   └── pipeline_manifest.json  # Complete/building/failed status
│   └── another_document/          # Entirely independent outputs
├── kg_construct/
│   ├── __init__.py
│   ├── chunker.py
│   ├── ontology_generator.py
│   └── kg_generator.py
├── kg_retrieve/
│   ├── __init__.py
│   ├── retrieval_config.py
│   ├── kg_retriever.py
│   └── qa_generator.py
├── main1.py
├── main2.py
├── requirements.txt
└── tests/
```

The `Data/` directories are generated when needed. The archive intentionally contains **no actual user PDFs, model weights, generated ontologies, or API keys**.

## 1. Setup

Use **Python 3.10+**. Open a terminal in the extracted `Project/` folder.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
$env:NVIDIA_API_KEY = "YOUR_NEW_NVIDIA_API_KEY"
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
export NVIDIA_API_KEY="YOUR_NEW_NVIDIA_API_KEY"
```

**Security:** An uploaded version of `qa_generator.py` contained an embedded NVIDIA API key. **Revoke/rotate that exposed key**, obtain a new one, and set it only via the environment variable above. The packaged version never embeds a key. Never commit a `.env` file or key to Git.

**Installation/model notes:** PyTorch installation varies by CPU/CUDA version. For a GPU build, use the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/) before installing the remaining packages. The first full run downloads MiniLM, GLiNER2, and NuExtract model weights from Hugging Face. Initial installation and local inference may use significant disk space and RAM; if GPU memory is limited, pass `--device cpu` (slower). `PDFs/` is just a convenient input directory: absolute input PDF paths also work. **This version processes text-based, English-language PDFs; scanned PDFs require OCR first.**

## 2. Generate the graph: `main1.py`

Copy `my_document.pdf` into `Project/PDFs/`, then:

```powershell
python main1.py
```

When prompted:

```text
Enter PDF name (example: my_document.pdf): my_document.pdf
```

You may also enter `my_document` (without `.pdf`) or an absolute path to an existing PDF. The program automatically builds `Data/my_document/` with all chunks, separate per-document ontology caches, ontology JSONs, graph JSON, GraphML export, per-chunk KG checkpoints, and a status manifest. It does **not** combine unrelated PDFs into one graph.

Non-interactive examples:

```powershell
python main1.py --pdf my_document.pdf
python main1.py --pdf "C:\Users\me\Documents\another_document.pdf" --device cpu
python main1.py --pdf my_document.pdf --chunk-size 1000 --chunk-overlap 200
python main1.py --pdf my_document.pdf --review
python main1.py --pdf my_document.pdf --refresh
python main1.py --pdf my_document.pdf --preview-only
python main1.py --help
```

- `--device auto` (default) selects CUDA if available, else CPU.
- `--review` adds the optional ontology-review LLM call; evidence-based refinement of sparse attributes is enabled by default. Use `--no-refine-sparse` to disable it.
- `--refresh` ignores existing ontology and KG checkpoints. Normal reruns reuse valid caches; a changed chunk or ontology invalidates its corresponding checkpoint automatically.
- `--preview-only` creates `ontology_candidates.json` using local NLP **without an NVIDIA call**; it does not create a final ontology or graph. Rerun normally to build the full graph.

`pipeline_manifest.json` is written as `building`, then `complete` only when the graph has been generated. If a build fails, the manifest says `failed`, so `main2.py` will not silently answer questions using an outdated graph. An interrupted model run can be resumed by running `main1.py` again with the same PDF and options. If the original PDF has been removed from `PDFs/`, `main1.py` can reuse that document's archived `Data/<name>/source.pdf`.

**Ontology limitations:** `ontology_coverage.json` measures local candidate/prompt coverage, **not proof of complete document-level concept recall**. Rich attributes are generated only when supported by the evidence. The original generator should not be interpreted as extracting every possible attribute or relation.

## 3. Ask questions: `main2.py`

After building at least one graph:

```powershell
python main2.py
```

Example session:

```text
Available PDFs: my_document.pdf, another_document.pdf
Enter PDF name for Q&A (example: my_document.pdf): my_document.pdf
[INFO] Graph loaded: ... nodes, ... edges.

Question: raw on
Raw retrieval display enabled.

Question: Who attended the event?
[RETRIEVAL] Searching the selected graph...
RAW KNOWLEDGE-GRAPH RETRIEVAL
... (JSON printed immediately, before the NVIDIA call)
[NVIDIA] Sending retrieved context to Nemotron...
ANSWER
... (generated answer)

Question: schema
... (entity and relation names and entity attributes)

Question: raw off
Question: exit
```

Non-interactive PDF selection (questions still interactive):

```powershell
python main2.py --pdf my_document.pdf --raw
python main2.py --pdf my_document --model nvidia/nemotron-3.5-lightning-30b-a3b
python main2.py --help
```

The retriever is **local**; only the final natural-language answer calls NVIDIA, and only if usable evidence was retrieved. Without `NVIDIA_API_KEY`, raw retrieval and `schema` inspection still work, but natural-language generation is unavailable. Remote answering has a bounded timeout and logs before the request rather than appearing frozen. If the graph contains no supporting facts, the application reports insufficient information instead of inventing a response. Retrieved nodes' attribute dictionaries are included in neighborhood results, including when an entity has no relationship edges.

## 4. Standalone module use

The original modules remain independently callable:

```powershell
python -m kg_construct.chunker PDFs/my_document.pdf --output Data/my_document/chunks
python -m kg_construct.ontology_generator --chunks Data/my_document/chunks --output Data/my_document/ontology.json --candidates-output Data/my_document/ontology_candidates.json --coverage-report Data/my_document/ontology_coverage.json --cache-dir Data/my_document/ontology_hybrid_cache
python -m kg_construct.kg_generator --chunks Data/my_document/chunks --ontology Data/my_document/ontology.json --output Data/my_document/knowledge_graph.json --graphml Data/my_document/knowledge_graph.graphml --cache-dir Data/my_document/kg_chunks
python -m kg_retrieve.kg_retriever Data/my_document/knowledge_graph.json
python -m kg_retrieve.qa_generator Data/my_document/knowledge_graph.json --raw
```

Using `main1.py` and `main2.py` is recommended because they select document-specific paths and track build status automatically.

## 5. Tests and troubleshooting

```powershell
python -m unittest discover -s tests -v
```

Tests cover PDF/folder selection, per-PDF output separation, the stage handoff using small mocked NLP/LLM components, missing evidence, and Q&A against a small fixture KG. They do **not** download large models or call NVIDIA. Full model inference/API integration requires your environment, model downloads, and a valid NVIDIA key.

- `PDF not found`: copy it to `Project/PDFs/`, supply the correct current-directory-relative path, or enter an absolute PDF path.
- `NVIDIA_API_KEY is not set`: configure it in the **same terminal session** before running `main1.py` or generating Q&A answers.
- spaCy `en_core_web_sm` missing: run `python -m spacy download en_core_web_sm` inside the virtual environment.
- Hugging Face model loading errors: check the internet connection, disk space, PyTorch/transformers compatibility, and CPU/CUDA settings.
- An image-only PDF produces no text: OCR it first, then run the pipeline on the resulting text-based PDF.
- A PDF has been edited: rerun `main1.py`; it compares the PDF SHA-256 digest and invalidates the generated state while preserving separately fingerprinted model checkpoints when possible.
- Graph Q&A finds entities but misses relevant facts: use `raw on` and `schema` to inspect the extracted relationships, attributes, and retrieval match candidates. The generation stage cannot recover facts absent from the PDF or unextracted by the local models.
