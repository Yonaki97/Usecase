import numpy as np

# COSINE SIMILARITY
def cosine_similarity(vec_a, vec_b):
    vec_a = np.asarray(
        vec_a,
        dtype=float
    )
    vec_b = np.asarray(
        vec_b,
        dtype=float
    )
    norm_a = np.linalg.norm(
        vec_a
    )
    norm_b = np.linalg.norm(
        vec_b
    )
    if norm_a == 0 or norm_b == 0:
        return 0.0
    similarity = np.dot(
        vec_a,
        vec_b
    ) / (
        norm_a * norm_b
    )
    return float(similarity)

# SIMILARITY SEARCH
def similarity_search(
    query_embedding,
    embedded_documents,
    top_k=5,
    document_types=None,
    min_similarity=0.0,
):
    # VALIDATION
    if top_k <= 0:
        raise ValueError(
            "top_k harus lebih besar dari 0."
        )
    if not 0 <= min_similarity <= 1:
        raise ValueError(
            "min_similarity harus berada "
            "antara 0 dan 1."
        )
    # RESULT CONTAINER
    results = []

    # LOOP KNOWLEDGE BASE
    for document in embedded_documents:
        metadata = document.get(
            "metadata",
            {}
        )

        # DOCUMENT TYPE
        document_type = metadata.get(
            "document_type"
        )
        if document_types is not None:
            if document_type not in document_types:
                continue

        # EMBEDDING
        document_embedding = document.get(
            "embedding"
        )
        if document_embedding is None:
            continue

        # COSINE SIMILARITY
        score = cosine_similarity(
            query_embedding,
            document_embedding
        )

        # MINIMUM THRESHOLD
        if score < min_similarity:
            continue

        # STORE CANDIDATE
        results.append({
    "chunk_id":
        document.get(
            "chunk_id"
        ),
    "text":
        document.get(
            "text",
            ""
        ),
    "embedding":
        document.get(
            "embedding"
        ),
    "similarity":
        round(
            score,
            4
        ),
    "metadata":
        metadata,
})

    # SORT
    results.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    # TOP-K
    return results[:top_k]