from pypdf import PdfReader
import os


# --------------------------------------------------
# 1. Extract text from PDF
# --------------------------------------------------

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    text = ""

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# --------------------------------------------------
# 2. Chunk the extracted text
# --------------------------------------------------

def chunk_text(text, chunk_size=1000, chunk_overlap=200):

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_text(text)

    return chunks


# --------------------------------------------------
# 3. Save chunks into folder
# --------------------------------------------------

def save_chunks(chunks, output_folder="chunks"):

    # Create folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    for i, chunk in enumerate(chunks, start=1):

        file_path = os.path.join(
            output_folder,
            f"chunk_{i:04d}.txt"
        )

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(chunk)

    print(f"Saved {len(chunks)} chunks inside '{output_folder}/'")


# --------------------------------------------------
# 4. Reusable pipeline entry point
# --------------------------------------------------

def process_pdf(pdf_path, output_folder, chunk_size=1000, chunk_overlap=200):
    """Extract text, split it, and save numbered chunks for one PDF.

    Returns the saved chunk count; callers can keep all files isolated by
    providing a document-specific ``output_folder``.
    """
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_size must be positive and chunk_overlap smaller")

    text = extract_text_from_pdf(pdf_path)
    print(f"[CHUNK] Extracted {len(text)} characters", flush=True)
    if not text.strip():
        raise ValueError(
            "This PDF contains no extractable text. Scanned/image-only PDFs "
            "need OCR before this pipeline can process them."
        )
    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise ValueError("Text extraction succeeded but produced no chunks")
    # Clear only old generated chunk files. Otherwise stale files from a
    # longer previous run could enter the ontology and knowledge graph.
    from pathlib import Path
    folder = Path(output_folder)
    folder.mkdir(parents=True, exist_ok=True)
    for stale in folder.glob("chunk_*.txt"):
        stale.unlink()
    save_chunks(chunks, output_folder=folder)
    return len(chunks)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Extract and chunk one PDF")
    parser.add_argument("pdf", help="Path to PDF")
    parser.add_argument("--output", default="chunks")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    args = parser.parse_args(argv)
    process_pdf(args.pdf, args.output, args.chunk_size, args.chunk_overlap)


if __name__ == "__main__":
    main()
