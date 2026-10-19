from s01_parser import parse_documents
from s02_cleaner import clean_documents
from s03_chunker import chunk_documents
from s04_metadata import extract_metadata
from s05_embedder import embed_documents
from save_database import save_database
 

def build_database():

    print("=" * 70)
    print("BUILD DATABASE")
    print("=" * 70)

    # ========================================================
    # 1. PARSE
    # ========================================================

    print("\n[1/6] Parsing...")

    documents = parse_documents()

    print(
        f"Dokumen: {len(documents)}"
    )

    # ========================================================
    # 2. CLEAN
    # ========================================================

    print("\n[2/6] Cleaning...")

    documents = clean_documents(
        documents
    )

    # ========================================================
    # 3. CHUNK
    # ========================================================

    print("\n[3/6] Chunking...")

    chunks = chunk_documents(
        documents
    )

    print(
        f"Chunk: {len(chunks)}"
    )

    # ========================================================
    # 4. METADATA
    # ========================================================

    print("\n[4/6] Metadata...")

    metadata_documents = extract_metadata(
        chunks
    )

    print(
        f"Metadata: "
        f"{len(metadata_documents)}"
    )

    # ========================================================
    # 5. EMBEDDING
    # ========================================================

    print("\n[5/6] Embedding...")

    embedded_documents = embed_documents(
        metadata_documents
    )

    print(
        f"Embedding: "
        f"{len(embedded_documents)}"
    )

    # ========================================================
    # 6. SAVE DATABASE
    # ========================================================

    print("\n[6/6] Saving database...")

    save_database(
        embedded_documents
    )

    print("\n" + "=" * 70)
    print("DATABASE SELESAI")
    print("=" * 70)


if __name__ == "__main__":
    build_database()