import ollama
import pickle
from pathlib import Path

# Configuration
EMBEDDING_MODEL = "qwen3-embedding:4b"
OLLAMA_HOST = "http://127.0.0.1:11434"

ollama_client = ollama.Client(
    host=OLLAMA_HOST
)

# Embedding File Path
def get_embedding_path():
    src_dir = Path(__file__).resolve().parent
    output_dir = src_dir / "output"
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Ubah nama model menjadi nama file
    model_name = (
        EMBEDDING_MODEL
        .replace(":", "_")
        .replace("/", "_")
    )

    output_path = (
        output_dir /
        f"embeddings_{model_name}.pkl"
    )
    return output_path

# Generate Single Embedding
def generate_embedding(text):
    if not text or not text.strip():
        raise ValueError(
            "embedding_text kosong."
        )
    print("\n" + "=" * 70)
    print("DEBUG EMBEDDING")
    print("=" * 70)
    print(
        "Panjang karakter:",
        len(text)
    )
    print(
        "Jumlah kata:",
        len(text.split())
    )
    print(
        "Preview:",
        text[:300]
    )
    try:
        response = ollama_client.embed(
            model=EMBEDDING_MODEL,
            input=text
        )
        embeddings = response.get(
            "embeddings"
        )
        if not embeddings:
            raise ValueError(
                "Ollama tidak mengembalikan embedding."
            )
        return embeddings[0]
    except Exception as e:
        print(
            "\n❌ GAGAL EMBEDDING"
        )
        print(
            "Error:",
            repr(e)
        )
        raise

# Generate All Embeddings
def embed_documents(metadata_documents):
    embedded_documents = []
    total = len(metadata_documents)
    for i, document in enumerate(metadata_documents):

        # Gunakan embedding_text untuk embedding
        embedding_text = document["embedding_text"]
        embedding = generate_embedding(
            embedding_text
        )
        # Simpan
        embedded_documents.append({
            "chunk_id":
                document["chunk_id"],
            # Teks asli
            "text":
                document["text"],
            # Teks yang digunakan untuk embedding
            "embedding_text":
                embedding_text,
            "embedding":
                embedding,
            "metadata":
                document["metadata"]
        })
        print(
            f"Embedding "
            f"{i + 1}/{total} selesai"
        )
    return embedded_documents

# Save Embeddings
def save_embeddings(embedded_documents):
    output_path = get_embedding_path()
    with open(
        output_path,
        "wb"
    ) as file:
        pickle.dump(
            embedded_documents,
            file
        )
    print(
        "\nEmbedding berhasil disimpan:"
    )
    print(output_path)

    return output_path

# Load Embeddings
def load_embeddings():
    output_path = get_embedding_path()
    if not output_path.exists():
        return None

    with open(
        output_path,
        "rb"
    ) as file:
        embedded_documents = pickle.load(
            file
        )
    print(
        "\nEmbedding berhasil dimuat:"
    )
    print(output_path)
    return embedded_documents

# Validate Embedding
def validate_embeddings(
    embedded_documents
):
    if not embedded_documents:
        print(
            "Tidak ada embedding untuk "
            "divalidasi."
        )
        return
    
    first_embedding = (
        embedded_documents[0]["embedding"]
    )
    print(
        "\n" + "=" * 70
    )
    print(
        "VALIDASI EMBEDDING"
    )
    print(
        "=" * 70
    )
    print(
        "Chunk ID:",
        embedded_documents[0]["chunk_id"]
    )
    print(
        "Dimensi embedding:",
        len(first_embedding)
    )
    print(
        "5 nilai pertama:",
        first_embedding[:5]
    )