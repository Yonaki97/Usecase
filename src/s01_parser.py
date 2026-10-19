from pathlib import Path
import pymupdf

# Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "01-raw"

# Parsing satu PDF
def parse_pdf(file_path):
    doc = pymupdf.open(file_path)
    pages = []

    for page_number, page in enumerate(doc):

        text = page.get_text("text")

        pages.append({
            "page_number": page_number + 1,
            "text": text
        })
    doc.close()
    return pages

# Parsing seluruh dokumen
def parse_documents():
    files = list(DATA_DIR.glob("*.pdf"))
    documents = []

    for file in files:

        pages = parse_pdf(file)

        documents.append({
            "file_name": file.name,
            "pages": pages
        })
    return documents

# Validasi hasil parsing
def validate_documents(documents):
    for document in documents:
        total_pages = len(document["pages"])
        pages_with_text = 0
        empty_pages = 0
        total_characters = 0

        for page in document["pages"]:

            text = page["text"].strip()
            if text:
                pages_with_text += 1
                total_characters += len(text)
            else:
                empty_pages += 1

        print("\n" + "=" * 60)
        print(f"Dokumen       : {document['file_name']}")
        print(f"Jumlah halaman: {total_pages}")
        print(f"Ada teks      : {pages_with_text}")
        print(f"Halaman kosong: {empty_pages}")
        print(f"Total karakter: {total_characters}")