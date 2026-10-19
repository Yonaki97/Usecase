
from s05_embedder import generate_embedding
from s06_similarity import similarity_search

from rubric import semua_sub_kriteria_flat

# CONFIGURATION
TOP_K_JUKNIS = 8
MIN_SIMILARITY_JUKNIS = 0.30

# BUILD JUKNIS QUERY
def build_juknis_query(sub_kriteria):
    evidence_to_find = sub_kriteria.get(
        "evidence_to_find",
        []
    )
    positive_indicators = sub_kriteria.get(
        "positive_indicators",
        []
    )
    negative_indicators = sub_kriteria.get(
        "negative_indicators",
        []
    )
    return f"""
Anda sedang mencari evidence dari Petunjuk Teknis
untuk kebutuhan assessment.

============================================================
ASSESSMENT SPECIFICATION
============================================================

Kategori:
{sub_kriteria.get("kategori_nama", "")}

Sub-kriteria:
{sub_kriteria.get("nama", "")}

Tujuan pemeriksaan:
{sub_kriteria.get("objective", "")}

Pertanyaan assessment:
{sub_kriteria.get("assessment_question", "")}

============================================================
EVIDENCE YANG DICARI
============================================================

{chr(10).join(
    "- " + x
    for x in evidence_to_find
)}

============================================================
INDIKATOR PEMENUHAN
============================================================

{chr(10).join(
    "- " + x
    for x in positive_indicators
)}

============================================================
INDIKATOR KETIDAKPEMENUHAN
============================================================

{chr(10).join(
    "- " + x
    for x in negative_indicators
)}

============================================================
KONDISI
============================================================

Kondisional:
{sub_kriteria.get("conditional", False)}

Kondisi:
{sub_kriteria.get("condition", "")}

============================================================
INSTRUKSI RETRIEVAL
============================================================

Cari bagian Juknis yang secara substantif
berhubungan dengan requirement yang dinilai.

Prioritaskan evidence yang:

1. Menjelaskan aturan atau ketentuan.

2. Menjelaskan prosedur atau tahapan.

3. Menjelaskan kewenangan.

4. Menjelaskan tugas dan tanggung jawab.

5. Menjelaskan pembagian fungsi.

6. Menjelaskan dokumen atau output.

7. Menjelaskan mekanisme pengendalian.

8. Menjelaskan monitoring atau evaluasi.

9. Menjelaskan SLA apabila relevan.

10. Menjelaskan metodologi apabila relevan.

11. Menjelaskan koordinasi, kalibrasi,
    atau rekonsiliasi apabila relevan.

Cari evidence yang dapat digunakan untuk
menjawab pertanyaan assessment.

Jangan melakukan assessment.

Jangan menentukan status.

Jangan menentukan score.

Jangan mencari hanya berdasarkan kesamaan kata.

Utamakan evidence substantif.
""".strip()

# CLEAN EVIDENCE FOR LLM
def clean_evidence_for_llm(evidence):
    if not isinstance(evidence, dict):
        return None
    metadata = evidence.get(
        "metadata",
        {}
    )
    return {
        "chunk_id":
            evidence.get("chunk_id"),
        "similarity":
            evidence.get(
                "similarity",
                0
            ),
        "text":
            evidence.get(
                "text",
                ""
            ),
        "document":
            metadata.get(
                "document_name"
            ),
        "document_type":
            metadata.get(
                "document_type"
            ),
        "bab":
            metadata.get(
                "bab"
            ),
        "bagian":
            metadata.get(
                "bagian"
            ),
        "pasal":
            metadata.get(
                "pasal"
            ),
        "title":
            metadata.get(
                "bab_title"
            ),
    }

# CLEAN EVIDENCE LIST
def clean_evidence_list(evidence_list):
    if not isinstance(
        evidence_list,
        list
    ):
        return []

    cleaned = []
    for evidence in evidence_list:
        item = clean_evidence_for_llm(
            evidence
        )
        if item is not None:
            cleaned.append(item)
    return cleaned

# FIND JUKNIS EVIDENCE
def find_juknis_evidence(
    sub_kriteria,
    juknis_embeddings,
    top_k=TOP_K_JUKNIS,
):
    if not juknis_embeddings:
        return []
    
    # QUERY
    query = build_juknis_query(
        sub_kriteria
    )

    # EMBEDDING
    query_embedding = generate_embedding(
        query
    )

    # SIMILARITY SEARCH
    results = similarity_search(
        query_embedding=query_embedding,
        embedded_documents=juknis_embeddings,
        top_k=top_k,
        min_similarity=MIN_SIMILARITY_JUKNIS,
    )
    return clean_evidence_list(
        results
    )

# BUILD EVIDENCE PACKAGE
def build_evidence_package(
    sub_kriteria,
    juknis_evidence,
    regulasi_evidence,
):
    return {
        "kode":
            sub_kriteria["kode"],
        "nama":
            sub_kriteria["nama"],
        "kategori_kode":
            sub_kriteria["kategori_kode"],
        "kategori_nama":
            sub_kriteria["kategori_nama"],
        "weight":
            sub_kriteria["weight"],

        # ASSESSMENT SPECIFICATION
        "objective":
            sub_kriteria.get(
                "objective",
                ""
            ),
        "assessment_question":
            sub_kriteria.get(
                "assessment_question",
                ""
            ),
        "evidence_to_find":
            sub_kriteria.get(
                "evidence_to_find",
                []
            ),
        "positive_indicators":
            sub_kriteria.get(
                "positive_indicators",
                []
            ),
        "negative_indicators":
            sub_kriteria.get(
                "negative_indicators",
                []
            ),
        "regulatory_focus":
            sub_kriteria.get(
                "regulatory_focus",
                []
            ),
        "conditional":
            sub_kriteria.get(
                "conditional",
                False
            ),
        "condition":
            sub_kriteria.get(
                "condition",
                ""
            ),

        # EVIDENCE
        "juknis":
            clean_evidence_list(
                juknis_evidence
            ),
        "regulasi":
            clean_evidence_list(
                regulasi_evidence
            ),
    }

# BUILD ALL EVIDENCE
def build_evidence_by_subcategory(
    juknis_embeddings,
    regulatory_references,
):
    if not isinstance(
        juknis_embeddings,
        list
    ):
        raise TypeError(
            "juknis_embeddings harus berupa list."
        )
    if not isinstance(
        regulatory_references,
        dict
    ):
        raise TypeError(
            "regulatory_references harus berupa dict."
        )
    evidence_by_subcategory = {}
    sub_kriteria_list = (
        semua_sub_kriteria_flat()
    )
    total = len(
        sub_kriteria_list
    )
    for index, sub in enumerate(
        sub_kriteria_list,
        start=1
    ):

        kode = sub["kode"]
        print(
            "\n" + "=" * 70
        )
        print(
            f"EVIDENCE RETRIEVAL "
            f"{index}/{total}: {kode}"
        )

        # JUKNIS
        juknis_evidence = find_juknis_evidence(
            sub_kriteria=sub,
            juknis_embeddings=(
                juknis_embeddings
            ),
            top_k=TOP_K_JUKNIS,
        )

        # REGULASI
        regulatory_reference = (
            regulatory_references.get(
                kode,
                {}
            )
        )
        regulasi_evidence = (
            regulatory_reference.get(
                "regulasi",
                []
            )
        )
        regulasi_evidence = (
            clean_evidence_list(
                regulasi_evidence
            )
        )

        # BUILD PACKAGE
        package = build_evidence_package(
            sub_kriteria=sub,
            juknis_evidence=(
                juknis_evidence
            ),
            regulasi_evidence=(
                regulasi_evidence
            ),
        )

        evidence_by_subcategory[kode] = package
        print(
            f"Juknis Evidence   : "
            f"{len(juknis_evidence)}"
        )
        print(
            f"Regulasi Evidence : "
            f"{len(regulasi_evidence)}"
        )

    return evidence_by_subcategory