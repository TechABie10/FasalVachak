import json
import chromadb
from chromadb.utils import embedding_functions
from kb.database import BASE_DIR, FULL_KB_DICT


def run_ingestion():
    master_kb = FULL_KB_DICT

    documents = []
    metadatas = []
    ids = []

    for disease_id, data in master_kb.items():
        bullets = data["symptoms_bullets"]

        # Flattened text is what gets embedded for similarity search
        documents.append(". ".join(b.rstrip(".") for b in bullets))

        metadatas.append({
            "disease_id": disease_id,
            "crop": data["crop"].capitalize(),
            "disease_name": data.get("disease_name", disease_id),
            # Keep the structured list around for the LLM step later,
            # JSON-encoded since Chroma metadata can't store lists directly
            "symptoms_bullets": json.dumps(bullets)
        })
        ids.append(disease_id)

    client = chromadb.PersistentClient(path=str(BASE_DIR / "rag" / "chroma_symptoms_db_raw"))
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.get_or_create_collection(
        name="agri_symptoms",
        embedding_function=embedding_func
    )

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Successfully indexed {len(ids)} diseases into ChromaDB.")


if __name__ == "__main__":
    run_ingestion()
