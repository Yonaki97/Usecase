from pathlib import Path
import pandas as pd


# ============================================================
# EXPORT CHUNKING
# ============================================================

def export_chunks_to_excel(chunks):

    # Tentukan lokasi src/
    src_dir = Path(__file__).resolve().parent

    # Folder output
    output_dir = src_dir / "output"
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Lokasi file
    output_path = output_dir / "chunk_preview.xlsx"

    # Buat DataFrame
    rows = []

    for i, chunk in enumerate(chunks):

        rows.append({

            # Preview index
            "chunk_index":
                i,

            # Document
            "file_name":
                chunk.get("file_name"),

            # Structure
            "section":
                chunk.get("section"),

            "bab":
                chunk.get("bab"),

            "bab_title":
                chunk.get("bab_title"),

            "bagian":
                chunk.get("bagian"),

            "subbagian":
                chunk.get("subbagian"),

            "article":
                chunk.get("article"),

            # Content
            "text":
                chunk.get("text"),

            "embedding_text":
                chunk.get("embedding_text")
        })

    # DataFrame
    df = pd.DataFrame(rows)

    # Export Excel
    df.to_excel(
        output_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("EXPORT CHUNKING")
    print("=" * 70)

    print("File berhasil dibuat:")
    print(output_path)

    return df


# ============================================================
# EXPORT METADATA
# ============================================================

def export_metadata_to_excel(
    metadata_documents
):

    # Tentukan lokasi src/
    src_dir = Path(__file__).resolve().parent

    # Folder output
    output_dir = src_dir / "output"
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Lokasi file
    output_path = (
        output_dir /
        "metadata_preview.xlsx"
    )

    # Buat rows
    rows = []

    for document in metadata_documents:

        metadata = document["metadata"]

        rows.append({

            # ====================================================
            # CHUNK ID
            # ====================================================

            "chunk_id":
                document.get("chunk_id"),

            # ====================================================
            # DOCUMENT IDENTITY
            # ====================================================

            "document_name":
                metadata.get("document_name"),

            "document_type":
                metadata.get("document_type"),

            "year":
                metadata.get("year"),

            "revision":
                metadata.get("revision"),

            # ====================================================
            # DOCUMENT STRUCTURE
            # ====================================================

            "section":
                metadata.get("section"),

            "bab":
                metadata.get("bab"),

            "bab_title":
                metadata.get("bab_title"),

            "bagian":
                metadata.get("bagian"),

            "subbagian":
                metadata.get("subbagian"),

            "pasal":
                metadata.get("pasal"),

            # ====================================================
            # CONTENT
            # ====================================================

            "text":
                document.get("text"),

            "embedding_text":
                document.get("embedding_text")
        })

    # DataFrame
    df = pd.DataFrame(rows)

    # Export Excel
    df.to_excel(
        output_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("EXPORT METADATA")
    print("=" * 70)

    print("File berhasil dibuat:")
    print(output_path)

    return df