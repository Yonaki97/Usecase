import json
import ollama

from rubric import semua_sub_kriteria_flat

# CONFIGURATION
LLM_MODEL = "qwen3.5:4b"
CONTEXT_WINDOW = 32000
MAX_OUTPUT_TOKENS = 2048 #2048

TEMPERATURE = 0

# ALLOWED STATUS
ALLOWED_STATUS = {
    "Sesuai",
    "Sebagian Sesuai",
    "Tidak Sesuai",
    "Tidak Ditemukan",
    "Tidak Dapat Dinilai",
}

STATUS_TO_LEVEL = {
    "Sesuai": 3,
    "Sebagian Sesuai": 2,
    "Tidak Sesuai": 1,
    "Tidak Ditemukan": 1,
    "Tidak Dapat Dinilai": None,
}

# NORMALIZE LEVEL
def normalize_assessment_level(level):
    if level is None:
        return None
    if isinstance(level, bool):
        return None
        
    try:
        value = float(level)
    except (TypeError, ValueError):
        return None
    
    if not value.is_integer():
        return None
    
    value = int(value)
    if value in (1, 2, 3):
        return value
    return None

# NORMALIZE STATUS
def normalize_status(status):
    if not isinstance(status, str):
        return "Tidak Dapat Dinilai"
    status = status.strip()

    if status in ALLOWED_STATUS:
        return status

    return "Tidak Dapat Dinilai"

# NORMALIZE EVIDENCE
def normalize_evidence(evidence):
    if not isinstance(evidence, list):
        return []
    normalized = []

    for item in evidence:
        if not isinstance(item, dict):
            continue
        chunk_id = item.get("chunk_id")
        text = item.get(
            "text",
            ""
        )
        similarity = item.get(
            "similarity"
        )
        metadata = item.get(
            "metadata",
            {}
        )

        if not isinstance(text, str):
            text = str(text)
        if not isinstance(metadata, dict):
            metadata = {}
        normalized.append({
            "chunk_id": chunk_id,
            "text": text,
            "similarity": similarity,
            "metadata": metadata,
        })

    return normalized

# PREPARE EVIDENCE FOR LLM
def prepare_evidence_for_llm(evidence):
    prepared = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        chunk_id = item.get("chunk_id")
        if not chunk_id:
            continue
        text = item.get(
            "text",
            ""
        )
        if not isinstance(text, str):
            text = str(text)
        prepared.append({
            "chunk_id": chunk_id,
            "text": text,
        })

    return prepared

# BUILD EVIDENCE LOOKUP
def build_evidence_lookup(evidence):
    lookup = {}
    if not isinstance(evidence, list):
        return lookup
    
    for item in evidence:
        if not isinstance(item, dict):
            continue
        chunk_id = item.get("chunk_id")
        if not chunk_id:
            continue
        lookup[chunk_id] = item

    return lookup

# EXTRACT CHUNK IDS FROM LLM
def extract_chunk_ids(evidence):
    ids = []
    if not isinstance(evidence, list):
        return ids

    for item in evidence:
        if isinstance(item, dict):
            chunk_id = item.get("chunk_id")
            if isinstance(chunk_id, str):
                chunk_id = chunk_id.strip()
                if chunk_id:
                    ids.append(chunk_id)
        elif isinstance(item, str):
            chunk_id = item.strip()
            if chunk_id:
                ids.append(chunk_id)
    return ids

# RECONSTRUCT EVIDENCE FROM ORIGINAL RETRIEVAL
def reconstruct_evidence(
    llm_evidence,
    original_evidence,
):
    lookup = build_evidence_lookup(
        original_evidence
    )
    chunk_ids = extract_chunk_ids(
        llm_evidence
    )
    reconstructed = []
    seen = set()
    for chunk_id in chunk_ids:
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        original = lookup.get(
            chunk_id
        )
        if original is None:
            continue
        reconstructed.append({
            "chunk_id":
                original.get("chunk_id"),
            "text":
                original.get("text", ""),
            "similarity":
                original.get("similarity"),
            "metadata":
                original.get("metadata", {}),
        })
    return reconstructed

# BUILD PROMPT
def build_assessment_prompt(
    sub_kriteria,
    evidence_juknis,
    evidence_regulasi,
):
    evidence_juknis_for_llm = (
        prepare_evidence_for_llm(
            evidence_juknis
        )
    )
    evidence_regulasi_for_llm = (
        prepare_evidence_for_llm(
            evidence_regulasi
        )
    )
    kode = sub_kriteria.get(
        "kode",
        ""
    )
    kategori = sub_kriteria.get(
        "kategori_nama",
        ""
    )
    nama = sub_kriteria.get(
        "nama",
        ""
    )

# Prompt
    return f"""
Anda adalah AI Assessment Engine untuk menilai
kecukupan Petunjuk Teknis (Juknis).

Anda BUKAN auditor yang bebas membuat kriteria sendiri.

Anda WAJIB mengikuti sub-kriteria yang diberikan.

============================================================
SUB-KRITERIA YANG HARUS DINILAI
============================================================

Kode:
{kode}

Kategori:
{kategori}

Requirement / Sub-kriteria:
{nama}

============================================================
EVIDENCE JUKNIS
============================================================

{json.dumps(
    evidence_juknis_for_llm,
    ensure_ascii=False,
    indent=2
)}

============================================================
EVIDENCE REGULASI
============================================================

{json.dumps(
    evidence_regulasi_for_llm,
    ensure_ascii=False,
    indent=2
)}

============================================================
TUJUAN ASSESSMENT
============================================================

Tentukan apakah Juknis memenuhi REQUIREMENT yang tertulis
dalam sub-kriteria.

Jangan mengganti requirement tersebut dengan requirement lain.

Assessment harus menjawab:

"Apakah isi Juknis memenuhi apa yang diminta oleh
sub-kriteria berdasarkan ketentuan regulasi yang tersedia?"

============================================================
URUTAN PENALARAN
============================================================

Lakukan penilaian dengan urutan berikut:

STEP 1
Identifikasi requirement eksplisit yang diminta oleh
sub-kriteria.

STEP 2
Identifikasi ketentuan regulasi yang relevan dari
evidence regulasi.

STEP 3
Identifikasi evidence Juknis yang menunjukkan
pemenuhan requirement.

STEP 4
Bandingkan evidence Juknis dengan requirement.

STEP 5
Periksa apakah ada evidence Juknis yang secara eksplisit
bertentangan dengan requirement.

STEP 6
Tentukan status berdasarkan aturan status di bawah.

============================================================
ATURAN PALING PENTING
============================================================

ATURAN 1:
JANGAN MEMBUAT REQUIREMENT BARU.

Hanya nilai requirement yang tertulis dalam sub-kriteria.

Jangan menambahkan persyaratan berdasarkan:
- pengetahuan umum;
- praktik audit;
- best practice;
- asumsi;
- interpretasi pribadi;
- prosedur organisasi;
- kebutuhan validasi;
- kebutuhan approval;
- kebutuhan review;
- kebutuhan monitoring;
- kebutuhan implementasi;
- kebutuhan dokumentasi tambahan.

Hal-hal tersebut hanya boleh dinilai jika secara eksplisit
diminta oleh sub-kriteria atau ditunjukkan sebagai ketentuan
yang wajib dalam evidence regulasi.

------------------------------------------------------------

ATURAN 2:
KETIADAAN BUKTI BUKAN OTOMATIS KETIDAKSESUAIAN.

Jika sub-kriteria hanya meminta A, dan evidence menunjukkan A,
maka A dapat dinilai sesuai.

Jangan mencari B, C, atau D jika B, C, dan D tidak diminta.

------------------------------------------------------------

ATURAN 3:
JANGAN MENGHUKUM JUKNIS KARENA TIDAK MENUNJUKKAN
PROSES YANG TIDAK DIMINTA.

Contoh:

Jika requirement hanya:
"Juknis memuat ketentuan sesuai PADG."

Dan evidence menunjukkan bahwa Juknis:
- mengacu pada PADG;
- menjabarkan ketentuan PADG;
- tidak bertentangan dengan PADG;

maka jangan menurunkan status hanya karena tidak ada
bukti mengenai:
- proses validasi;
- approval;
- review;
- monitoring;
- pengesahan;
- implementasi;
- atau proses lain yang tidak diminta.

------------------------------------------------------------

ATURAN 4:
EVIDENCE REGULASI MENENTUKAN ATURAN YANG BERLAKU.

Evidence regulasi bukan bukti bahwa Juknis otomatis sesuai.

Namun evidence regulasi harus digunakan untuk memahami
apa yang diwajibkan oleh regulasi.

Evidence Juknis kemudian dibandingkan dengan kewajiban tersebut.

------------------------------------------------------------

ATURAN 5:
JIKA JUKNIS SECARA EKSPLISIT MEMENUHI REQUIREMENT,
DAN TIDAK ADA EVIDENCE KONTRADIKSI,
MAKA NILAI "SESUAI".

Jangan menurunkan menjadi "Sebagian Sesuai" hanya karena
masih mungkin terdapat informasi lain yang tidak ditemukan.

------------------------------------------------------------

ATURAN 6:
"SEBAGIAN SESUAI" SANGAT TERBATAS.

Gunakan "Sebagian Sesuai" HANYA jika:

1. sub-kriteria memiliki lebih dari satu requirement
eksplisit; DAN

2. sebagian requirement tersebut terpenuhi; DAN

3. sebagian requirement eksplisit lainnya tidak terpenuhi.

Jangan menggunakan "Sebagian Sesuai" hanya karena:
- evidence tidak lengkap;
- evidence sedikit;
- similarity rendah;
- tidak ditemukan validasi;
- tidak ditemukan approval;
- tidak ditemukan review;
- LLM merasa kurang yakin;
- ada kemungkinan requirement tambahan;
- dokumen tidak menjelaskan hal yang tidak diminta.

------------------------------------------------------------

ATURAN 7:
"TIDAK SESUAI" MEMERLUKAN BUKTI KONTRADIKSI.

Gunakan "Tidak Sesuai" apabila evidence Juknis menunjukkan
bahwa Juknis bertentangan dengan requirement yang dinilai.

Jangan menggunakan "Tidak Sesuai" hanya karena evidence
pemenuhan tidak ditemukan.

------------------------------------------------------------

ATURAN 8:
"TIDAK DITEMUKAN" BERARTI REQUIREMENT TIDAK DAPAT DIBUKTIKAN
DARI EVIDENCE JUKNIS YANG DIBERIKAN.

Gunakan status ini apabila:
- requirement jelas;
- tidak ada evidence Juknis yang menunjukkan pemenuhan;
- dan tidak ada evidence Juknis yang menunjukkan
  pertentangan secara eksplisit.

------------------------------------------------------------

ATURAN 9:
"TIDAK DAPAT DINILAI" HARUS SANGAT JARANG.

Gunakan hanya apabila:
- evidence saling bertentangan dan tidak dapat diselesaikan;
atau
- evidence yang tersedia tidak memungkinkan perbandingan
  yang valid sama sekali.

Jangan menggunakan "Tidak Dapat Dinilai" hanya karena
LLM merasa tidak yakin.

============================================================
STATUS
============================================================

Pilih tepat SATU:

"Sesuai"
"Sebagian Sesuai"
"Tidak Sesuai"
"Tidak Ditemukan"
"Tidak Dapat Dinilai"

------------------------------------------------------------

SESUAI:

Gunakan jika evidence Juknis memenuhi requirement eksplisit
dari sub-kriteria berdasarkan ketentuan regulasi yang relevan,
dan tidak terdapat kontradiksi.

------------------------------------------------------------

SEBAGIAN SESUAI:

Gunakan hanya jika terdapat beberapa requirement eksplisit
dan hanya sebagian yang terpenuhi.

------------------------------------------------------------

TIDAK SESUAI:

Gunakan jika evidence Juknis secara eksplisit bertentangan
dengan requirement.

------------------------------------------------------------

TIDAK DITEMUKAN:

Gunakan jika requirement dapat dinilai tetapi evidence Juknis
yang diberikan tidak menunjukkan pemenuhan requirement.

------------------------------------------------------------

TIDAK DAPAT DINILAI:

Gunakan hanya jika perbandingan tidak mungkin dilakukan
secara valid atau evidence saling bertentangan.

============================================================
ASSESSMENT LEVEL
============================================================

Sesuai = 3

Sebagian Sesuai = 2

Tidak Sesuai = 1

Tidak Ditemukan = 1

Tidak Dapat Dinilai = null

Assessment level harus mengikuti status.

============================================================
SIMILARITY
============================================================

Similarity HANYA menunjukkan relevansi hasil retrieval.

Similarity BUKAN:
- assessment score;
- confidence score;
- compliance score;
- dasar langsung untuk menentukan status.

Jangan mengatakan:
"similarity rendah sehingga tidak sesuai."

============================================================
EVIDENCE
============================================================

Untuk evidence_juknis dan evidence_regulasi,
KEMBALIKAN HANYA chunk_id yang benar-benar menjadi
dasar keputusan.

Jangan menulis ulang text evidence.

Jangan membuat chunk_id baru.

Chunk_id yang dikembalikan HARUS berasal dari evidence
yang diberikan.

============================================================
ALASAN
============================================================

Alasan harus:

1. menjelaskan requirement yang dinilai;
2. menyebut evidence Juknis yang relevan;
3. menyebut evidence regulasi yang relevan jika tersedia;
4. menjelaskan hubungan antara keduanya;
5. menjelaskan alasan status.

Jangan membuat requirement baru dalam alasan.

Jangan meminta bukti tambahan yang tidak diwajibkan
oleh sub-kriteria.

============================================================
CONTOH POLA PENILAIAN
============================================================

CONTOH 1 - SESUAI

Requirement:
"Juknis menjabarkan ketentuan PADG."

Evidence regulasi:
PADG menjadi dasar yang harus dijabarkan dalam Juknis.

Evidence Juknis:
Juknis secara eksplisit menjabarkan ketentuan PADG.

Tidak ada kontradiksi.

Kesimpulan:
Sesuai.

------------------------------------------------------------

CONTOH 2 - TIDAK SESUAI

Requirement:
"Juknis menetapkan batas waktu 3 hari."

Evidence regulasi:
Batas waktu wajib = 3 hari.

Evidence Juknis:
Juknis menetapkan batas waktu = 7 hari.

Kesimpulan:
Tidak Sesuai.

------------------------------------------------------------

CONTOH 3 - SEBAGIAN SESUAI

Requirement eksplisit:
A wajib.
B wajib.
C wajib.

Evidence Juknis:
A terpenuhi.
B terpenuhi.
C tidak terpenuhi.

Kesimpulan:
Sebagian Sesuai.

------------------------------------------------------------

CONTOH 4 - TIDAK DITEMUKAN

Requirement:
"Juknis mengatur mekanisme X."

Evidence Juknis:
Tidak terdapat evidence yang menunjukkan mekanisme X.

Tidak terdapat evidence yang menunjukkan pertentangan.

Kesimpulan:
Tidak Ditemukan.

------------------------------------------------------------

CONTOH 5 - JANGAN MEMBUAT REQUIREMENT

Requirement:
"Juknis sesuai dengan PADG."

Evidence Juknis:
Juknis mengacu dan menjabarkan ketentuan PADG.

Evidence tambahan mengenai proses validasi tidak tersedia.

KESIMPULAN YANG BENAR:
Sesuai.

JANGAN:
Sebagian Sesuai karena tidak ada bukti validasi.

============================================================
BATASAN
============================================================

Gunakan HANYA evidence yang diberikan.

Jangan menggunakan pengetahuan eksternal.

Jangan mengarang isi dokumen.

Jangan mengarang chunk_id.

Jangan menggunakan similarity untuk menentukan status.

Jangan menilai berdasarkan jumlah evidence.

Jangan menilai berdasarkan panjang evidence.

Jangan menilai berdasarkan panjang dokumen.

Jangan mencari requirement tambahan.

============================================================
OUTPUT
============================================================

Keluarkan JSON SAJA.

Format:

{{
    "kode": "{kode}",
    "status": "Sesuai",
    "assessment_level": 3,
    "evidence_juknis": [
        "chunk_id"
    ],
    "evidence_regulasi": [
        "chunk_id"
    ],
    "alasan": "Alasan singkat dan berbasis evidence."
}}

PENTING:

- status harus salah satu dari status yang diizinkan;
- assessment_level harus mengikuti status;
- evidence harus berupa chunk_id;
- jangan mengembalikan text evidence;
- jangan membuat chunk_id;
- alasan harus berdasarkan evidence;
- jangan menambahkan requirement baru.

Keluarkan JSON SAJA.
""".strip()

# NORMALIZE LLM RESULT
def normalize_result(
    parsed,
    sub_kriteria,
    evidence_juknis,
    evidence_regulasi,
):
    kode = sub_kriteria["kode"]
    if not isinstance(parsed, dict):
        return {
            "kode": kode,
            "status": "Tidak Dapat Dinilai",
            "assessment_level": None,
            "evidence_juknis": [],
            "evidence_regulasi": [],
            "alasan": "Output LLM tidak valid.",
        }

    # STATUS
    status = normalize_status(
        parsed.get("status")
    )

    # LEVEL
    level = STATUS_TO_LEVEL[
        status
    ]

    # EVIDENCE
    llm_evidence_juknis = (
        parsed.get(
            "evidence_juknis",
            []
        )
    )
    llm_evidence_regulasi = (
        parsed.get(
            "evidence_regulasi",
            []
        )
    )
    reconstructed_juknis = (
        reconstruct_evidence(
            llm_evidence=llm_evidence_juknis,
            original_evidence=evidence_juknis,
        )
    )
    reconstructed_regulasi = (
        reconstruct_evidence(
            llm_evidence=llm_evidence_regulasi,
            original_evidence=evidence_regulasi,
        )
    )

    # ALASAN
    alasan = parsed.get(
        "alasan",
        ""
    )
    if not isinstance(alasan, str):
        alasan = str(alasan)
    alasan = alasan.strip()
    if not alasan:
        alasan = (
            "LLM tidak memberikan alasan assessment."
        )
    return {
        "kode": kode,
        "status": status,
        "assessment_level": level,
        "evidence_juknis":
            reconstructed_juknis,
        "evidence_regulasi":
            reconstructed_regulasi,
        "alasan":
            alasan,
    }

# ASSESS ONE SUBCATEGORY
def assess_subcategory(
    sub_kriteria,
    evidence_juknis,
    evidence_regulasi,
):
    # SAFETY NORMALIZATION
    if not isinstance(
        evidence_juknis,
        list
    ):
        evidence_juknis = []
    if not isinstance(
        evidence_regulasi,
        list
    ):
        evidence_regulasi = []

    # BUILD PROMPT
    prompt = build_assessment_prompt(
        sub_kriteria=sub_kriteria,
        evidence_juknis=evidence_juknis,
        evidence_regulasi=evidence_regulasi,
    )

    system_prompt = """
Anda adalah AI Assessment Engine untuk menilai
kecukupan Petunjuk Teknis.

PRINSIP UTAMA:

1. Nilai HANYA requirement yang tertulis dalam
   sub-kriteria.

2. Jangan membuat requirement tambahan.

3. Jangan menggunakan pengetahuan eksternal.

4. Gunakan hanya evidence yang diberikan.

5. Evidence regulasi menjelaskan aturan yang menjadi
   dasar pembanding.

6. Evidence Juknis menunjukkan apakah requirement
   dipenuhi.

7. Jika Juknis memenuhi requirement secara substantif
   dan tidak ada kontradiksi, pilih "Sesuai".

8. Jangan menurunkan status karena tidak adanya bukti
   atas hal yang tidak diminta.

9. "Sebagian Sesuai" hanya untuk sebagian requirement
   eksplisit yang terpenuhi.

10. "Tidak Sesuai" memerlukan evidence kontradiksi.

11. "Tidak Ditemukan" berarti requirement dapat dinilai
    tetapi evidence Juknis yang tersedia tidak menunjukkan
    pemenuhannya.

12. "Tidak Dapat Dinilai" hanya digunakan jika penilaian
    benar-benar tidak dapat dilakukan atau evidence
    saling bertentangan.

Output JSON valid saja.
"""

    # CALL OLLAMA
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],

            think=False,
            stream=False,
            format="json",
            options={
                "temperature":
                    TEMPERATURE,
                "num_ctx":
                    CONTEXT_WINDOW,
                "num_predict":
                    MAX_OUTPUT_TOKENS,
            },
        )
    except Exception as exc:
        print(
            "\n❌ GAGAL LLM ASSESSMENT"
        )
        print(
            f"Error: {repr(exc)}"
        )
        return {
            "kode":
                sub_kriteria["kode"],
            "status":
                "Tidak Dapat Dinilai",
            "assessment_level":
                None,
            "evidence_juknis":
                [],
            "evidence_regulasi":
                [],
            "alasan":
                f"LLM tidak dapat dihubungi: {exc}",
        }

    # EXTRACT CONTENT
    content = (
        response
        .get("message", {})
        .get("content", "")
    )
    print("\nLLM RAW OUTPUT:")
    print(content)

    # PARSE JSON
    try:
        parsed = json.loads(
            content
        )
    except (
        json.JSONDecodeError,
        TypeError,
    ):
        parsed = None

    # NORMALIZE
    return normalize_result(
        parsed=parsed,
        sub_kriteria=
            sub_kriteria,
        evidence_juknis=
            evidence_juknis,
        evidence_regulasi=
            evidence_regulasi,
    )

# ASSESS ALL SUBCATEGORIES
def assess_all_subcategories(
    evidence_by_subcategory,
):
    if not isinstance(
        evidence_by_subcategory,
        dict
    ):
        raise TypeError(
            "evidence_by_subcategory harus berupa dict."
        )
    results = {}
    sub_kriteria_list = (
        semua_sub_kriteria_flat()
    )
    total = len(
        sub_kriteria_list
    )
    for index, sub in enumerate(
        sub_kriteria_list,
        start=1,
    ):
        kode = sub["kode"]
        print(
            "\n" + "=" * 70
        )
        print(
            f"LLM ASSESSMENT "
            f"{index}/{total}: {kode}"
        )
        evidence = (
            evidence_by_subcategory.get(
                kode,
                {}
            )
        )
        if not isinstance(
            evidence,
            dict
        ):
            evidence = {}
        evidence_juknis = (
            evidence.get(
                "juknis",
                []
            )
        )
        evidence_regulasi = (
            evidence.get(
                "regulasi",
                []
            )
        )

        if not isinstance(
            evidence_juknis,
            list
        ):
            evidence_juknis = []

        if not isinstance(
            evidence_regulasi,
            list
        ):
            evidence_regulasi = []

        # ASSESS
        result = assess_subcategory(
            sub_kriteria=
                sub,
            evidence_juknis=
                evidence_juknis,
            evidence_regulasi=
                evidence_regulasi,
        )

        # ADD RUBRIC METADATA
        result["nama"] = (
            sub.get(
                "nama",
                ""
            )
        )
        result["kategori_kode"] = (
            sub.get(
                "kategori_kode"
            )
        )
        result["kategori_nama"] = (
            sub.get(
                "kategori_nama"
            )
        )
        result["weight"] = (
            sub.get(
                "weight"
            )
        )
        
        # SAVE
        results[kode] = result

        # DEBUG OUTPUT
        print(
            f"Status           : "
            f"{result['status']}"
        )
        print(
            f"Assessment Level : "
            f"{result['assessment_level']}"
        )
        print(
            f"Juknis Evidence  : "
            f"{len(result['evidence_juknis'])}"
        )
        print(
            f"Regulasi Evidence: "
            f"{len(result['evidence_regulasi'])}"
        )

    return results