import os
import re
import hashlib


# ============================================================
# DOCUMENT TYPE
# ============================================================

def extract_document_type(file_name):
    """
    Menentukan tipe dokumen berdasarkan nama file.

    Mendukung:
        PBI
        PADG
        PADGI
        JUKNIS
        UNKNOWN

    Penting:
    Juknis tidak harus bernama JUKNIS_xxx.pdf.
    """

    if not file_name:
        return "UNKNOWN"

    base_name = os.path.basename(
        str(file_name)
    )

    stem = os.path.splitext(
        base_name
    )[0]

    normalized = re.sub(
        r"[\s_\-]+",
        " ",
        stem.upper()
    ).strip()

    # ========================================================
    # PADGI harus diperiksa sebelum PADG
    # ========================================================

    if re.search(
        r"\bPADGI\b",
        normalized
    ):
        return "PADGI"

    if re.search(
        r"\bPADG\b",
        normalized
    ):
        return "PADG"

    if re.search(
        r"\bPBI\b",
        normalized
    ):
        return "PBI"

    # ========================================================
    # JUKNIS
    # ========================================================

    if (
        "JUKNIS" in normalized
        or "PETUNJUK TEKNIS" in normalized
        or "PETUNJUK-TEKNIS" in stem.upper()
        or "PETUNJUK_TEKNIS" in stem.upper()
    ):
        return "JUKNIS"

    return "UNKNOWN"


# ============================================================
# YEAR
# ============================================================

def extract_year(file_name):
    """
    Mengambil tahun dari nama file.

    Prioritas:
        20xx
        19xx

    Contoh:
        PBI_23_2021.pdf -> 2021
        PADG_12_2023.pdf -> 2023
        Juknis_2025.pdf -> 2025
    """

    if not file_name:
        return None

    stem = os.path.splitext(
        os.path.basename(
            str(file_name)
        )
    )[0]

    # ========================================================
    # PRIORITAS TAHUN 4 DIGIT
    # ========================================================

    years = re.findall(
        r"\b(?:19|20)\d{2}\b",
        stem
    )

    if years:
        return int(years[-1])

    # ========================================================
    # FALLBACK 2 DIGIT
    # ========================================================

    numbers = re.findall(
        r"\d+",
        stem
    )

    if not numbers:
        return None

    for number in reversed(numbers):

        if len(number) == 2:

            year = int(number)

            # Konteks regulasi modern
            if 0 <= year <= 99:
                return 2000 + year

    return None


# ============================================================
# REVISION
# ============================================================

def extract_revision(file_name):
    """
    Mengambil informasi revisi dari nama file.

    Contoh:
        Rev
        Rev1
        Rev2
        Revisi
        Revisi1
    """

    if not file_name:
        return None

    stem = os.path.splitext(
        os.path.basename(
            str(file_name)
        )
    )[0]

    match = re.search(
        r"(?i)\b(?:rev|revisi)"
        r"(?:[\s_\-]*(\d+))?\b",
        stem
    )

    if not match:
        return None

    revision_number = match.group(1)

    if revision_number:
        return f"Rev{revision_number}"

    return "Rev"


# ============================================================
# SAFE FILE ID
# ============================================================

def normalize_file_stem(file_name):
    """
    Mengubah nama file menjadi identifier aman.
    """

    if not file_name:
        return "unknown"

    stem = os.path.splitext(
        os.path.basename(
            str(file_name)
        )
    )[0]

    stem = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        stem
    )

    stem = re.sub(
        r"_+",
        "_",
        stem
    )

    return stem.strip("_") or "unknown"


# ============================================================
# GENERATE CHUNK ID
# ============================================================

def generate_chunk_id(
    file_name,
    index
):
    """
    ID deterministik dan relatif aman.

    Contoh:
        Petunjuk_Teknis_KKS_0001
    """

    base_name = normalize_file_stem(
        file_name
    )

    return f"{base_name}_{index:04d}"


# ============================================================
# DOCUMENT HASH
# ============================================================

def generate_document_id(file_name):
    """
    Identifier pendek untuk membedakan dokumen.
    """

    if not file_name:
        return "doc_unknown"

    raw = str(
        os.path.basename(
            file_name
        )
    ).lower()

    digest = hashlib.sha1(
        raw.encode("utf-8")
    ).hexdigest()[:10]

    return f"doc_{digest}"


# ============================================================
# EXTRACT METADATA
# ============================================================

def extract_metadata(chunks):
    """
    Mengubah hasil chunker menjadi format yang digunakan
    embedding + similarity + retrieval.

    Output konsisten:

    {
        "chunk_id": ...,
        "text": ...,
        "embedding_text": ...,
        "metadata": {
            "document_name": ...,
            "document_type": ...,
            "year": ...,
            "revision": ...,
            "section": ...,
            "bab": ...,
            "bab_title": ...,
            "bagian": ...,
            "subbagian": ...,
            "pasal": ...
        }
    }
    """

    if not isinstance(
        chunks,
        list
    ):
        raise TypeError(
            "chunks harus berupa list."
        )

    metadata_documents = []

    for index, chunk in enumerate(
        chunks
    ):

        if not isinstance(
            chunk,
            dict
        ):
            continue

        file_name = chunk.get(
            "file_name"
        )

        # ====================================================
        # CORE TEXT
        # ====================================================

        text = chunk.get(
            "text",
            ""
        )

        embedding_text = chunk.get(
            "embedding_text",
            ""
        )

        # Jangan biarkan embedding_text kosong
        if not embedding_text:
            embedding_text = text

        # ====================================================
        # DOCUMENT INFO
        # ====================================================

        document_type = extract_document_type(
            file_name
        )

        year = extract_year(
            file_name
        )

        revision = extract_revision(
            file_name
        )

        document_id = generate_document_id(
            file_name
        )

        chunk_id = generate_chunk_id(
            file_name,
            index
        )

        # ====================================================
        # METADATA
        # ====================================================

        metadata = {
            "document_id":
                document_id,

            "document_name":
                file_name,

            "document_type":
                document_type,

            "year":
                year,

            "revision":
                revision,

            "section":
                chunk.get(
                    "section"
                ),

            "bab":
                chunk.get(
                    "bab"
                ),

            "bab_title":
                chunk.get(
                    "bab_title"
                ),

            "bagian":
                chunk.get(
                    "bagian"
                ),

            "subbagian":
                chunk.get(
                    "subbagian"
                ),

            # Nama standar yang digunakan retrieval
            "pasal":
                chunk.get(
                    "article"
                ),

            # Alias untuk kompatibilitas
            "article":
                chunk.get(
                    "article"
                ),

            "chunk_part":
                chunk.get(
                    "chunk_part",
                    1
                ),

            "chunk_total":
                chunk.get(
                    "chunk_total",
                    1
                ),
        }

        metadata_documents.append({
            "chunk_id":
                chunk_id,

            "text":
                text,

            "embedding_text":
                embedding_text,

            "metadata":
                metadata
        })

    return metadata_documents