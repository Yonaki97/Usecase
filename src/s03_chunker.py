import re


# ============================================================
# CONFIGURATION
# ============================================================

# Safety net.
# Chunk utama tetap berdasarkan struktur dokumen.
MAX_CHUNK_CHARS = 5000

# Overlap hanya digunakan jika satu paragraf/ayat terlalu panjang.
CHUNK_OVERLAP_CHARS = 300


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text):
    """
    Normalisasi ringan hasil OCR / text extraction.

    Prinsip:
    - tidak mengubah isi substantif;
    - mempertahankan struktur baris;
    - menghapus whitespace berlebih;
    - mengurangi blank line berulang.
    """

    if not text:
        return ""

    text = str(text)

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Normalize non-breaking space
    text = text.replace("\u00a0", " ")

    # Hapus trailing whitespace
    lines = [
        line.rstrip()
        for line in text.split("\n")
    ]

    normalized_lines = []

    previous_blank = False

    for line in lines:
        line = line.rstrip()

        if not line.strip():
            if previous_blank:
                continue

            normalized_lines.append("")
            previous_blank = True

        else:
            normalized_lines.append(line)
            previous_blank = False

    return "\n".join(normalized_lines).strip()


# ============================================================
# STRUCTURE DETECTION
# ============================================================

# Roman numeral
ROMAN_PATTERN = r"[IVXLCDM]+"

# Kata ordinal Bagian yang umum dipakai regulasi
BAGIAN_WORD_PATTERN = (
    r"Kesatu|Kedua|Ketiga|Keempat|Kelima|"
    r"Keenam|Ketujuh|Kedelapan|Kesembilan|Kesepuluh|"
    r"Kesebelas|Keduabelas"
)


def detect_structure(text):
    """
    Mendeteksi struktur dokumen regulasi dan Juknis.

    Struktur utama:

        BAB I
        BAB II

        Bagian Kesatu
        Bagian Kedua

        Pasal 1
        Pasal 2

        A. Latar Belakang
        B. Dasar Hukum

        1. Tujuan
        2. Ruang Lingkup

        1.1 ...
        1.2 ...

        PENJELASAN

    Hanya heading yang benar-benar berada di awal baris
    yang dianggap sebagai struktur.
    """

    if not text:
        return []

    patterns = [

        # ====================================================
        # BAB
        # ====================================================
        (
            "bab",
            re.compile(
                rf"(?im)^[ \t]*(BAB\s+{ROMAN_PATTERN}"
                rf"(?:[ \t]+[^\n]+)?)[ \t]*$"
            )
        ),

        # ====================================================
        # BAGIAN
        # ====================================================
        (
            "bagian",
            re.compile(
                rf"(?im)^[ \t]*(Bagian\s+"
                rf"(?:{BAGIAN_WORD_PATTERN}|{ROMAN_PATTERN})"
                rf"(?:[ \t]+[^\n]+)?)[ \t]*$"
            )
        ),

        # ====================================================
        # PASAL
        # ====================================================
        (
            "pasal",
            re.compile(
                r"(?im)^[ \t]*(Pasal\s+\d+"
                r"(?:[ \t]+[^\n]+)?)[ \t]*$"
            )
        ),

        # ====================================================
        # LETTER SECTION
        # A. Latar Belakang
        # B. Dasar Hukum
        # ====================================================
        (
            "subbagian",
            re.compile(
                r"^[ \t]*([A-Z]\.\s+[^\n]+)[ \t]*$",
                re.MULTILINE
            )
        ),

        # ====================================================
        # NUMBERED SECTION
        # 1. Tujuan
        # 2. Ruang Lingkup
        # ====================================================
        (
            "subbagian_number",
            re.compile(
                r"^[ \t]*(\d+\.\s+[^\n]+)[ \t]*$",
                re.MULTILINE
            )
        ),

        # ====================================================
        # DECIMAL SECTION
        # 1.1 Tujuan
        # 1.2 Ruang Lingkup
        # ====================================================
        (
            "subbagian_decimal",
            re.compile(
                r"^[ \t]*(\d+\.\d+(?:\.\d+)*\s+[^\n]+)[ \t]*$",
                re.MULTILINE
            )
        ),

        # ====================================================
        # PENJELASAN
        # ====================================================
        (
            "penjelasan",
            re.compile(
                r"(?im)^[ \t]*(PENJELASAN)[ \t]*$"
            )
        ),
    ]

    matches = []

    for structure_type, pattern in patterns:

        for match in pattern.finditer(text):

            heading = match.group(1).strip()

            matches.append({
                "heading": heading,
                "type": structure_type,
                "start": match.start(),
                "end": match.end(),
            })

    # ========================================================
    # SORT BY POSITION
    # ========================================================

    matches.sort(
        key=lambda item: (
            item["start"],
            item["end"]
        )
    )

    # ========================================================
    # REMOVE DUPLICATES / OVERLAPPING MATCHES
    # ========================================================

    cleaned = []

    for match in matches:

        if not cleaned:
            cleaned.append(match)
            continue

        previous = cleaned[-1]

        # Exact duplicate
        if (
            match["start"] == previous["start"]
            and match["end"] == previous["end"]
        ):
            continue

        # Overlap
        if match["start"] < previous["end"]:

            # Prefer more specific / stronger structure
            priority = {
                "bab": 100,
                "bagian": 90,
                "pasal": 80,
                "penjelasan": 70,
                "subbagian_decimal": 60,
                "subbagian_number": 50,
                "subbagian": 40,
            }

            if priority.get(
                match["type"],
                0
            ) > priority.get(
                previous["type"],
                0
            ):
                cleaned[-1] = match

            continue

        cleaned.append(match)

    return cleaned


# ============================================================
# EXTRACT BAB TITLE
# ============================================================

def extract_bab_title(
    text,
    bab_structure,
    structures
):
    """
    Mengambil judul BAB yang berada tepat setelah heading BAB.

    Contoh:

        BAB I
        KETENTUAN UMUM

        Pasal 1

    hasil:

        KETENTUAN UMUM

    Tidak mengambil isi Pasal.
    """

    start = bab_structure["end"]

    # Cari struktur berikutnya
    next_structure = None

    for structure in structures:

        if structure["start"] <= bab_structure["start"]:
            continue

        next_structure = structure
        break

    if next_structure is None:
        end = len(text)
    else:
        end = next_structure["start"]

    title_area = text[start:end].strip()

    if not title_area:
        return None

    lines = [
        line.strip()
        for line in title_area.splitlines()
        if line.strip()
    ]

    if not lines:
        return None

    title_lines = []

    for line in lines:

        # Jangan masukkan heading lain
        if re.match(
            r"^(?:"
            r"Pasal\s+\d+"
            r"|Bagian\s+"
            r"|[A-Z]\.\s+"
            r"|\d+\.\s+"
            r"|\d+\.\d+(?:\.\d+)*\s+"
            r")",
            line,
            re.IGNORECASE
        ):
            break

        title_lines.append(line)

        # Judul BAB umumnya pendek
        if len(title_lines) >= 3:
            break

    if not title_lines:
        return None

    return " ".join(title_lines)


# ============================================================
# TITLE HELPERS
# ============================================================

def extract_subbagian_title(heading):
    if not heading:
        return None

    return heading.strip()


def extract_bagian_title(heading):
    if not heading:
        return None

    return heading.strip()


# ============================================================
# BUILD EMBEDDING TEXT
# ============================================================

def build_embedding_text(
    bab=None,
    bab_title=None,
    bagian=None,
    subbagian=None,
    article=None,
    section=None,
    text=""
):
    """
    Membuat teks khusus untuk embedding.

    Hierarchy:

        SECTION
        BAB
        BAB TITLE
        BAGIAN
        SUBBAGIAN
        PASAL
        CONTENT

    Metadata tetap disimpan terpisah.
    """

    context_parts = []

    if section:
        context_parts.append(
            f"Bagian Dokumen: {section}"
        )

    if bab:
        context_parts.append(
            f"{bab}"
        )

    if bab_title:
        context_parts.append(
            f"Judul BAB: {bab_title}"
        )

    if bagian:
        context_parts.append(
            f"{bagian}"
        )

    if subbagian:
        context_parts.append(
            f"{subbagian}"
        )

    if article:
        context_parts.append(
            f"{article}"
        )

    if text:
        context_parts.append(
            text
        )

    return "\n".join(
        part.strip()
        for part in context_parts
        if part and part.strip()
    )


# ============================================================
# SPLIT LONG TEXT
# ============================================================

def split_long_text(
    text,
    max_chars=MAX_CHUNK_CHARS,
    overlap=CHUNK_OVERLAP_CHARS
):
    """
    Safety-net.

    Prioritas:
    1. paragraf
    2. kalimat
    3. karakter

    Ini TIDAK digunakan untuk membentuk chunk utama.
    """

    if not text:
        return []

    text = text.strip()

    if len(text) <= max_chars:
        return [text]

    # ========================================================
    # SPLIT PARAGRAPH
    # ========================================================

    paragraphs = [
        p.strip()
        for p in re.split(
            r"\n\s*\n+",
            text
        )
        if p.strip()
    ]

    chunks = []
    current = ""

    for paragraph in paragraphs:

        candidate = (
            f"{current}\n\n{paragraph}"
            if current
            else paragraph
        )

        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())
            current = ""

        # ====================================================
        # PARAGRAPH MASIH TERLALU PANJANG
        # ====================================================

        if len(paragraph) > max_chars:

            # Coba pecah berdasarkan kalimat
            sentences = re.split(
                r"(?<=[.!?])\s+",
                paragraph
            )

            sentence_buffer = ""

            for sentence in sentences:

                sentence = sentence.strip()

                if not sentence:
                    continue

                candidate_sentence = (
                    f"{sentence_buffer} {sentence}"
                    if sentence_buffer
                    else sentence
                )

                if len(candidate_sentence) <= max_chars:
                    sentence_buffer = candidate_sentence
                    continue

                if sentence_buffer:
                    chunks.append(
                        sentence_buffer.strip()
                    )

                # Kalimat sendiri terlalu panjang
                if len(sentence) > max_chars:

                    start = 0

                    while start < len(sentence):

                        end = min(
                            start + max_chars,
                            len(sentence)
                        )

                        piece = sentence[
                            start:end
                        ].strip()

                        if piece:
                            chunks.append(piece)

                        if end >= len(sentence):
                            break

                        start = max(
                            0,
                            end - overlap
                        )

                    sentence_buffer = ""

                else:
                    sentence_buffer = sentence

            if sentence_buffer:
                chunks.append(
                    sentence_buffer.strip()
                )

        else:
            current = paragraph

    if current:
        chunks.append(current.strip())

    return chunks


# ============================================================
# CHUNK ONE DOCUMENT
# ============================================================

def chunk_document(document):
    """
    Hierarchical chunking.

    Regulasi:

        BAB
          └── Bagian
                └── Pasal

    Juknis:

        BAB
          └── A. / B. / C.
                └── content

    Jika Pasal tersedia:
        unit utama = Pasal

    Jika tidak ada Pasal:
        unit utama = subbagian

    Jika tidak ada struktur:
        fallback = dokumen dengan safety split.
    """

    pages = document.get(
        "pages",
        []
    )

    # ========================================================
    # COMBINE PAGES
    # ========================================================

    page_texts = []

    for page in pages:

        if not isinstance(page, dict):
            continue

        page_text = page.get(
            "text",
            ""
        )

        if page_text:
            page_texts.append(
                str(page_text)
            )

    text = "\n".join(page_texts)

    text = normalize_text(text)

    if not text:
        return []

    # ========================================================
    # DETECT STRUCTURE
    # ========================================================

    structures = detect_structure(text)

    chunks = []

    # ========================================================
    # FALLBACK
    # ========================================================

    if not structures:

        parts = split_long_text(text)

        for index, part in enumerate(
            parts,
            start=1
        ):

            chunks.append({
                "file_name": document.get(
                    "file_name"
                ),
                "section": "main",
                "bab": None,
                "bab_title": None,
                "bagian": None,
                "subbagian": None,
                "article": None,
                "text": part,
                "embedding_text": build_embedding_text(
                    section="main",
                    text=part
                ),
                "chunk_part": index,
                "chunk_total": len(parts),
            })

        return chunks

    # ========================================================
    # CURRENT HIERARCHY
    # ========================================================

    current_section = "main"

    current_bab = None
    current_bab_title = None
    current_bagian = None
    current_subbagian = None
    current_article = None

    current_start = None

    # ========================================================
    # SAVE CURRENT CHUNK
    # ========================================================

    def save_current_chunk(end_position):

        nonlocal current_start

        if current_start is None:
            return

        if end_position <= current_start:
            current_start = None
            return

        raw_text = text[
            current_start:end_position
        ].strip()

        current_start = None

        if not raw_text:
            return

        # ====================================================
        # REMOVE HEADING-LIKE EMPTY CONTENT
        # ====================================================

        text_parts = split_long_text(
            raw_text
        )

        if not text_parts:
            return

        total_parts = len(text_parts)

        for part_index, text_part in enumerate(
            text_parts,
            start=1
        ):

            embedding_text = build_embedding_text(
                section=current_section,
                bab=current_bab,
                bab_title=current_bab_title,
                bagian=current_bagian,
                subbagian=current_subbagian,
                article=current_article,
                text=text_part
            )

            chunk = {
                "file_name": document.get(
                    "file_name"
                ),

                "section": current_section,

                "bab": current_bab,

                "bab_title": current_bab_title,

                "bagian": current_bagian,

                "subbagian": current_subbagian,

                "article": current_article,

                "text": text_part,

                "embedding_text": embedding_text,

                "chunk_part": part_index,

                "chunk_total": total_parts,
            }

            chunks.append(chunk)

    # ========================================================
    # DOCUMENT OPENING
    # ========================================================

    first_structure = structures[0]

    if first_structure["start"] > 0:

        opening_text = text[
            :first_structure["start"]
        ].strip()

        if opening_text:

            opening_parts = split_long_text(
                opening_text
            )

            for part_index, part in enumerate(
                opening_parts,
                start=1
            ):

                chunks.append({
                    "file_name": document.get(
                        "file_name"
                    ),
                    "section": "opening",
                    "bab": None,
                    "bab_title": None,
                    "bagian": None,
                    "subbagian": None,
                    "article": "Pembukaan",
                    "text": part,
                    "embedding_text": build_embedding_text(
                        section="opening",
                        article="Pembukaan",
                        text=part
                    ),
                    "chunk_part": part_index,
                    "chunk_total": len(
                        opening_parts
                    )
                })

    # ========================================================
    # PROCESS STRUCTURES
    # ========================================================

    for structure in structures:

        heading = structure["heading"]
        structure_type = structure["type"]

        # ====================================================
        # SAVE CONTENT BEFORE NEW STRUCTURE
        # ====================================================

        save_current_chunk(
            structure["start"]
        )

        # ====================================================
        # PENJELASAN
        # ====================================================

        if structure_type == "penjelasan":

            current_section = "explanation"

            current_bab = None
            current_bab_title = None
            current_bagian = None
            current_subbagian = None
            current_article = None

            current_start = structure["end"]

            continue

        # ====================================================
        # BAB
        # ====================================================

        if structure_type == "bab":

            current_section = "main"

            current_bab = heading

            current_bab_title = extract_bab_title(
                text,
                structure,
                structures
            )

            current_bagian = None
            current_subbagian = None
            current_article = None

            # Tidak langsung membuat chunk.
            current_start = None

            continue

        # ====================================================
        # BAGIAN
        # ====================================================

        if structure_type == "bagian":

            current_bagian = extract_bagian_title(
                heading
            )

            current_subbagian = None
            current_article = None

            current_start = structure["end"]

            continue

        # ====================================================
        # SUBBAGIAN
        # ====================================================

        if structure_type in {
            "subbagian",
            "subbagian_number",
            "subbagian_decimal"
        }:

            current_subbagian = extract_subbagian_title(
                heading
            )

            current_article = None

            current_start = structure["end"]

            continue

        # ====================================================
        # PASAL
        # ====================================================

        if structure_type == "pasal":

            current_article = heading

            current_start = structure["end"]

            continue

    # ========================================================
    # SAVE LAST CONTENT
    # ========================================================

    save_current_chunk(
        len(text)
    )

    return chunks


# ============================================================
# CHUNK ALL DOCUMENTS
# ============================================================

def chunk_documents(documents):
    """
    Melakukan hierarchical chunking terhadap seluruh dokumen.
    """

    if not isinstance(
        documents,
        list
    ):
        raise TypeError(
            "documents harus berupa list."
        )

    all_chunks = []

    for document in documents:

        if not isinstance(
            document,
            dict
        ):
            continue

        document_chunks = chunk_document(
            document
        )

        all_chunks.extend(
            document_chunks
        )

    return all_chunks