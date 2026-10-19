import pickle
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent

DATABASE_DIR = PROJECT_DIR / "data" / "04-database"

DATABASE_FILE = (
    DATABASE_DIR /
    "embeddings_qwen3-embedding_4b.pkl"
)

def save_database(embedded_documents):

    DATABASE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        DATABASE_FILE,
        "wb"
    ) as file:

        pickle.dump(
            embedded_documents,
            file
        )

    print("\n" + "=" * 70)
    print("DATABASE BERHASIL DISIMPAN")
    print("=" * 70)

    print(
        "Jumlah embedding:",
        len(embedded_documents)
    )

    print(
        "File:",
        DATABASE_FILE
    )

    return DATABASE_FILE