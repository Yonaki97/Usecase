from pathlib import Path

from s01_parser import parse_pdf
from s02_cleaner import clean_documents
from s03_chunker import chunk_documents
from s04_metadata import extract_metadata

from s05_embedder import (
    embed_documents,
    load_embeddings,
)

from assesment_retriever import (
    retrieve_for_assessment
)

from assesment import (
    build_evidence_by_subcategory
)

from s07_llm import (
    assess_all_subcategories
)

from scoring import (
    hitung_skor_akhir
)

from llm_judge import (
    judge_all_subcategories
)


# ============================================================
# CONFIGURATION
# ============================================================

TOP_K_REGULASI = 5


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_assessment(
    pdf_path,
    file_name=None,
    progress_callback=None,
    enable_llm_judge=True,
    judge_model=None
):

    # ========================================================
    # PROGRESS
    # ========================================================

    def progress(value, message):

        if progress_callback:
            progress_callback(
                value,
                message
            )


    # ========================================================
    # 1. PARSING
    # ========================================================

    progress(
        0.05,
        "Membaca dokumen Juknis..."
    )

    pages = parse_pdf(pdf_path)

    if not pages:
        raise ValueError(
            "PDF tidak memiliki halaman yang dapat dibaca."
        )

    document_name = (
        file_name
        if file_name
        else Path(pdf_path).name
    )

    documents = [
        {
            "file_name": document_name,
            "pages": pages
        }
    ]

    print("\n" + "=" * 80)
    print("1. PARSING JUKNIS")
    print("=" * 80)

    print(
        "Nama dokumen :",
        document_name
    )

    print(
        "Jumlah halaman:",
        len(pages)
    )


    # ========================================================
    # 2. CLEANING
    # ========================================================

    progress(
        0.12,
        "Membersihkan dokumen Juknis..."
    )

    cleaned_documents = clean_documents(
        documents
    )

    if not cleaned_documents:
        raise ValueError(
            "Hasil cleaning Juknis kosong."
        )


    # ========================================================
    # 3. CHUNKING
    # ========================================================

    progress(
        0.20,
        "Menganalisis struktur Juknis..."
    )

    chunks = chunk_documents(
        cleaned_documents
    )

    print("\n" + "=" * 80)
    print("3. CHUNKING JUKNIS")
    print("=" * 80)

    print(
        "Jumlah chunk:",
        len(chunks)
    )

    if not chunks:
        raise ValueError(
            "Tidak ada chunk Juknis yang dapat dianalisis."
        )


    # ========================================================
    # 4. METADATA
    # ========================================================

    progress(
        0.27,
        "Menyiapkan metadata Juknis..."
    )

    metadata_documents = extract_metadata(
        chunks
    )

    if not metadata_documents:
        raise ValueError(
            "Metadata Juknis kosong."
        )

    print("\n" + "=" * 80)
    print("4. METADATA JUKNIS")
    print("=" * 80)

    document_types = {}

    for item in metadata_documents:

        metadata = item.get(
            "metadata",
            {}
        )

        document_type = metadata.get(
            "document_type",
            "UNKNOWN"
        )

        document_types[document_type] = (
            document_types.get(
                document_type,
                0
            ) + 1
        )

    print(
        "Distribusi document_type:",
        document_types
    )


    # ========================================================
    # 5. EMBEDDING JUKNIS
    # ========================================================

    progress(
        0.38,
        "Membuat embedding Juknis..."
    )

    juknis_embeddings = embed_documents(
        metadata_documents
    )

    if not juknis_embeddings:
        raise ValueError(
            "Embedding Juknis kosong."
        )

    print("\n" + "=" * 80)
    print("5. EMBEDDING JUKNIS")
    print("=" * 80)

    print(
        "Jumlah embedding Juknis:",
        len(juknis_embeddings)
    )


    # ========================================================
    # 6. LOAD REGULATORY KNOWLEDGE BASE
    # ========================================================

    progress(
        0.48,
        "Memuat knowledge base regulasi..."
    )

    regulatory_embeddings = load_embeddings()

    if regulatory_embeddings is None:
        raise FileNotFoundError(
            "File embedding knowledge base regulasi "
            "tidak ditemukan."
        )

    if not regulatory_embeddings:
        raise ValueError(
            "Embedding knowledge base regulasi kosong."
        )

    print("\n" + "=" * 80)
    print("6. REGULATORY KNOWLEDGE BASE")
    print("=" * 80)

    print(
        "Jumlah embedding regulasi:",
        len(regulatory_embeddings)
    )

    regulatory_types = {}

    for item in regulatory_embeddings:

        metadata = item.get(
            "metadata",
            {}
        )

        document_type = metadata.get(
            "document_type",
            "UNKNOWN"
        )

        regulatory_types[document_type] = (
            regulatory_types.get(
                document_type,
                0
            ) + 1
        )

    print(
        "Distribusi document_type:",
        regulatory_types
    )


    # ========================================================
    # 7. REGULATORY RETRIEVAL
    # ========================================================

    progress(
        0.58,
        "Mencari ketentuan regulasi yang relevan..."
    )

    print("\n" + "=" * 100)
    print("7. REGULATORY RETRIEVAL")
    print("=" * 100)

    assessment_references = (
        retrieve_for_assessment(

            regulatory_embeddings=
                regulatory_embeddings,

            top_k_regulasi=
                TOP_K_REGULASI
        )
    )

    if not assessment_references:
        raise ValueError(
            "Regulatory retrieval menghasilkan data kosong."
        )


    # ========================================================
    # 8. DEBUG REGULATORY RETRIEVAL
    # ========================================================

    print("\n" + "=" * 100)
    print("8. HASIL REGULATORY RETRIEVAL")
    print("=" * 100)

    for kode, result in (
        assessment_references.items()
    ):

        print("\n" + "#" * 100)

        print(
            "SUB-KRITERIA :",
            kode
        )

        print(
            "NAMA         :",
            result.get("nama", "")
        )

        print(
            "KATEGORI     :",
            result.get("kategori_nama", "")
        )

        regulasi_results = (
            result.get(
                "regulasi",
                []
            )
        )

        print(
            "Jumlah regulasi evidence:",
            len(regulasi_results)
        )

        for rank, item in enumerate(
            regulasi_results,
            start=1
        ):

            metadata = item.get(
                "metadata",
                {}
            )

            print(
                f"\nRank #{rank}"
            )

            # PENTING:
            # S06 menggunakan key "similarity"
            print(
                "Similarity :",
                item.get(
                    "similarity",
                    0
                )
            )

            print(
                "Chunk ID   :",
                item.get("chunk_id")
            )

            print(
                "Document   :",
                metadata.get(
                    "document_name"
                )
            )

            print(
                "Type       :",
                metadata.get(
                    "document_type"
                )
            )

            print(
                "BAB        :",
                metadata.get("bab")
            )

            print(
                "Bagian     :",
                metadata.get("bagian")
            )

            print(
                "Pasal      :",
                metadata.get("pasal")
            )

            text = item.get(
                "text",
                ""
            )

            print(
                "Text       :",
                text[:500].replace(
                    "\n",
                    " "
                )
            )


    # ========================================================
    # 9. BUILD EVIDENCE PACKAGE
    # ========================================================

    progress(
        0.70,
        "Menyiapkan evidence Juknis dan regulasi..."
    )

    print("\n" + "=" * 100)
    print("9. BUILD EVIDENCE PACKAGE")
    print("=" * 100)

    evidence_by_subcategory = (
        build_evidence_by_subcategory(

            juknis_embeddings=
                juknis_embeddings,

            regulatory_references=
                assessment_references
        )
    )

    if not evidence_by_subcategory:
        raise ValueError(
            "Evidence package kosong."
        )


    # ========================================================
    # 10. DEBUG EVIDENCE PACKAGE
    # ========================================================

    print("\n" + "=" * 100)
    print("10. EVIDENCE PACKAGE")
    print("=" * 100)

    for kode, evidence in (
        evidence_by_subcategory.items()
    ):

        juknis_count = len(
            evidence.get(
                "juknis",
                []
            )
        )

        regulasi_count = len(
            evidence.get(
                "regulasi",
                []
            )
        )

        print(
            f"{kode} | "
            f"Juknis = {juknis_count} | "
            f"Regulasi = {regulasi_count}"
        )


    # ========================================================
    # 11. LLM ASSESSMENT
    # ========================================================

    progress(
        0.78,
        "Melakukan assessment dengan LLM..."
    )

    print("\n" + "=" * 100)
    print("11. LLM ASSESSMENT")
    print("=" * 100)

    assessment_results = (
        assess_all_subcategories(
            evidence_by_subcategory=
                evidence_by_subcategory
        )
    )

    if not assessment_results:
        raise ValueError(
            "LLM assessment menghasilkan data kosong."
        )


    # ========================================================
    # 12. DEBUG LLM ASSESSMENT
    # ========================================================

    print("\n" + "=" * 100)
    print("12. HASIL LLM ASSESSMENT")
    print("=" * 100)

    for kode, result in (
        assessment_results.items()
    ):

        print("\n" + "-" * 80)

        print(
            kode
        )

        print(
            "Nama             :",
            result.get("nama", "")
        )

        print(
            "Status           :",
            result.get("status")
        )

        print(
            "Assessment Level :",
            result.get("assessment_level")
        )

        print(
            "Weight           :",
            result.get("weight")
        )

        print(
            "Juknis Evidence  :",
            len(
                result.get(
                    "evidence_juknis",
                    []
                )
            )
        )

        print(
            "Regulasi Evidence:",
            len(
                result.get(
                    "evidence_regulasi",
                    []
                )
            )
        )

        print(
            "Alasan           :",
            result.get(
                "alasan",
                ""
            )
        )


    # ========================================================
    # 13. SCORING
    # ========================================================

    progress(
        0.92,
        "Menghitung assessment score..."
    )

    print("\n" + "=" * 100)
    print("13. SCORING")
    print("=" * 100)

    score_result = hitung_skor_akhir(
        assessment_results
    )

    # Secondary evaluation only: this never changes the Qwen score.
    judge_results = None
    if enable_llm_judge:
        progress(0.96, "Memvalidasi output Qwen dengan Qwen judge...")
        judge_kwargs = {"model": judge_model} if judge_model else {}
        judge_results = judge_all_subcategories(
            evidence_by_subcategory,
            assessment_results,
            **judge_kwargs
        )

    print(
        "Raw Score:",
        score_result["total_score"]
    )

    print(
        "Max Score:",
        score_result["max_score"]
    )

    print(
        "Assessment Score:",
        score_result["percentage"]
    )

    print(
        "Classification:",
        score_result["classification"]
    )


    # ========================================================
    # 14. SELESAI
    # ========================================================

    progress(
        1.0,
        "Assessment selesai."
    )

    print("\n" + "=" * 100)
    print("ASSESSMENT SELESAI")
    print("=" * 100)


    # ========================================================
    # FINAL RESULT
    # ========================================================

    return {

        "document_name":
            document_name,

        "total_score":
            score_result["total_score"],

        "max_score":
            score_result["max_score"],

        "percentage":
            score_result["percentage"],

        "classification":
            score_result["classification"],

        "assessment_results":
            assessment_results,

        "assessment_references":
            assessment_references,

        "evidence_by_subcategory":
            evidence_by_subcategory,

        "llm_judge_results":
            judge_results,
    }
