import chromadb
from chromadb.utils import embedding_functions
from kb.database import BASE_DIR, FULL_KB_DICT

# 1. Load the Master Knowledge Base into memory once
MASTER_KB = FULL_KB_DICT

# 2. Connect to the existing ChromaDB collection
client = chromadb.PersistentClient(path=str(BASE_DIR / "rag" / "chroma_symptoms_db_raw"))
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = client.get_collection(
    name="agri_symptoms",
    embedding_function=embedding_func
)


def get_candidates(crop: str, symptom_text: str, n_results: int = 3) -> str:
    """
    Filters candidates by crop and queries ChromaDB with the farmer's initial symptom text.
    Returns a pre-formatted string ready to be injected into DISAMBIGUATION_PROMPT.
    """
    # Normalize crop name ("rice" -> "Rice", "cotton" -> "Cotton")
    crop_filter = crop.strip().capitalize()

    results = collection.query(
        query_texts=[symptom_text],
        n_results=n_results,
        where={"crop": crop_filter}
    )

    candidate_lines = []
    matched_ids = results["ids"][0]
    # flattened symptoms_bullets text, same as what was embedded
    matched_docs = results["documents"][0]

    for idx, (disease_id, symptom_detail) in enumerate(zip(matched_ids, matched_docs), start=1):
        record = MASTER_KB.get(disease_id, {})
        disease_name = record.get("disease_name", disease_id)

        # Matches your required prompt format: "1. {name}: {symptoms}"
        candidate_lines.append(f"{idx}. {disease_name}: {symptom_detail}")

    return "\n".join(candidate_lines)


def get_disease_record(disease_identifier: str) -> dict:
    """
    O(1) flat dictionary lookup for Phase 2 (Management & Follow-up).
    Resolves by disease_id or disease_name.
    """
    # Direct ID match
    if disease_identifier in MASTER_KB:
        return MASTER_KB[disease_identifier]

    # Name-based fallback match (case-insensitive)
    for record in MASTER_KB.values():
        if record.get("disease_name", "").lower() == disease_identifier.lower():
            return record

    return {}
