import re

def clean_text(text):
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return text

def clean_documents(documents):
    cleaned_documents = []
    for document in documents:
        cleaned_pages = []

        for page in document["pages"]:
            cleaned_text = clean_text(page["text"])
            cleaned_pages.append({
                "page_number": page["page_number"],
                "text": cleaned_text
            })

        cleaned_documents.append({
            "file_name": document["file_name"],
            "pages": cleaned_pages
        })

    return cleaned_documents

