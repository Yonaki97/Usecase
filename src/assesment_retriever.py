
from s05_embedder import generate_embedding
from s06_similarity import similarity_search

from rubric import semua_sub_kriteria_flat

# CONFIGURATION
TOP_K_REGULASI = 8
MIN_SIMILARITY_REGULASI = 0.30

# BUILD RUBRIC CONTEXT
def build_rubric_context(sub_kriteria):

    return f"""
Kategori:
{sub_kriteria.get("kategori_nama", "")}

Sub-kriteria:
{sub_kriteria.get("nama", "")}

Tujuan pemeriksaan:
{sub_kriteria.get("objective", "")}

Pertanyaan assessment:
{sub_kriteria.get("assessment_question", "")}

Evidence yang perlu dicari:
{chr(10).join("- " + x for x in sub_kriteria.get("evidence_to_find", []))}

Fokus regulasi:
{chr(10).join("- " + x for x in sub_kriteria.get("regulatory_focus", []))}

Kriteria positif:
{chr(10).join("- " + x for x in sub_kriteria.get("positive_indicators", []))}

Kriteria negatif:
{chr(10).join("- " + x for x in sub_kriteria.get("negative_indicators", []))}

Kondisional:
{sub_kriteria.get("conditional", False)}

Kondisi:
{sub_kriteria.get("condition", "")}
""".strip()

# BUILD REGULATORY QUERY

def build_regulasi_query(sub_kriteria):
    context = build_rubric_context(sub_kriteria)

    return f"""
Anda sedang mencari regulatory evidence
untuk melakukan assessment kecukupan Juknis.

Tujuan retrieval adalah menemukan ketentuan
PBI, PADG, atau PADG Intern yang dapat menjadi
dasar pembanding terhadap Juknis.

============================================================
ASSESSMENT SPECIFICATION
============================================================

{context}

============================================================
INSTRUKSI RETRIEVAL
============================================================

Cari ketentuan regulasi yang:

1. Secara substantif mengatur objek yang dinilai.

2. Dapat digunakan sebagai dasar pembanding
   terhadap Juknis.

3. Memuat norma, kewajiban, kewenangan,
   prosedur, tahapan, tanggung jawab,
   persyaratan, pengendalian, atau mekanisme
   yang relevan dengan sub-kriteria.

4. Jika assessment berkaitan dengan kewenangan,
   prioritaskan ketentuan yang secara eksplisit
   memberikan atau membatasi kewenangan.

5. Jika assessment berkaitan dengan proses,
   prioritaskan ketentuan yang menjelaskan
   tahapan atau mekanisme proses.

6. Jika assessment berkaitan dengan pembagian
   tugas, prioritaskan ketentuan mengenai
   peran, tanggung jawab, kewenangan,
   pemisahan fungsi, atau approval.

7. Jika assessment berkaitan dengan dokumen,
   prioritaskan ketentuan mengenai dokumen,
   record, laporan, atau bukti pelaksanaan.

8. Jika assessment berkaitan dengan monitoring
   atau evaluasi, prioritaskan ketentuan yang
   mengatur monitoring, evaluasi, review,
   pelaporan, atau tindak lanjut.

9. Jika assessment bersifat kondisional,
   cari juga ketentuan yang dapat membantu
   menentukan apakah kondisi tersebut berlaku.

Jangan mencari hanya berdasarkan kesamaan kata.

Utamakan semantic relevance dan fungsi ketentuan
sebagai dasar pembanding assessment.

Jangan melakukan assessment.

Jangan menentukan apakah Juknis Sesuai atau
Tidak Sesuai.

Tugas Anda hanya menghasilkan query yang
memaksimalkan kemungkinan menemukan regulatory
evidence yang relevan.
""".strip()

# RETRIEVE ONE SUB-KRITERIA
def retrieve_for_sub_kriteria(
    sub_kriteria,
    regulatory_embeddings,
    top_k_regulasi=TOP_K_REGULASI,
):
    kode = sub_kriteria["kode"]

    # BUILD QUERY
    query = build_regulasi_query(sub_kriteria)
    # EMBEDDING QUERY
    query_embedding = generate_embedding(query)

    # SIMILARITY SEARCH
    regulasi_results = similarity_search(
        query_embedding=query_embedding,
        embedded_documents=(regulatory_embeddings),
        top_k=top_k_regulasi,
        document_types=[
            "PBI",
            "PADG",
            "PADGI",
        ],
        min_similarity=(MIN_SIMILARITY_REGULASI),
    )
    # RETURN
    return {
        "kode": kode,
        "nama": sub_kriteria["nama"],
        "kategori_kode": sub_kriteria["kategori_kode"],
        "kategori_nama": sub_kriteria["kategori_nama"],
        "weight": sub_kriteria["weight"],
        # Simpan assessment specification
        # agar dapat digunakan untuk debugging.
        "objective": sub_kriteria.get("objective", ""),
        "assessment_question": sub_kriteria.get("assessment_question", ""),
        "regulatory_focus": sub_kriteria.get("regulatory_focus", []),
        "rubric_context": build_rubric_context(sub_kriteria),
        "query_regulasi": query,
        "regulasi": regulasi_results,
    }

# RETRIEVE ALL REGULATIONS

def retrieve_for_assessment(
    regulatory_embeddings,
    top_k_regulasi=TOP_K_REGULASI,
):
    if not isinstance(regulatory_embeddings, list):
        raise TypeError("regulatory_embeddings harus berupa list.")

    sub_kriteria_list = semua_sub_kriteria_flat()
    assessment_references = {}
    total = len(sub_kriteria_list)
    for index, sub_kriteria in enumerate(sub_kriteria_list, start=1):
        kode = sub_kriteria["kode"]

        print("\n" + "=" * 70)

        print(f"REGULATORY RETRIEVAL {index}/{total}: {kode}")

        result = retrieve_for_sub_kriteria(
            sub_kriteria=sub_kriteria,
            regulatory_embeddings=(regulatory_embeddings),
            top_k_regulasi=(top_k_regulasi),
        )

        assessment_references[kode] = result
        print(f"Regulasi candidate: {len(result['regulasi'])}")

    return assessment_references